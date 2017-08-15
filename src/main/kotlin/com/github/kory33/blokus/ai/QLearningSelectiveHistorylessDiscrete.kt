package com.github.kory33.blokus.ai

import com.github.kory33.blokus.environment.space.SelectiveDiscreteSpace
import org.deeplearning4j.rl4j.learning.Learning
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.policy.EpsGreedy
import org.deeplearning4j.rl4j.policy.Policy
import org.deeplearning4j.rl4j.util.Constants
import org.deeplearning4j.rl4j.util.DataManager
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

abstract class QLearningSelectiveHistorylessDiscrete<O : ISelectiveObservation, AS : SelectiveDiscreteSpace>
        (private val mdp: MDP<O, Int, AS>,
         dqn: IDQN, conf: QLearning.QLConfiguration,
         private val dataManager: DataManager,
         epsilonNbStep: Int)
    : QLearning<O, Int, AS>(conf) {

    private val configuration: QLearning.QLConfiguration = conf
    override fun getConfiguration(): QLConfiguration = configuration

    override fun getDataManager(): DataManager = dataManager

    override fun getMdp(): MDP<O, Int, AS> = mdp

    private var currentDQN: IDQN = dqn
    override fun getCurrentDQN(): IDQN = currentDQN

    private var policy: SelectiveDQNPolicy<O, AS>
    override fun getPolicy(): Policy<O, Int> = policy

    private var egPolicy: EpsGreedy<O, Int, AS>
    override fun getEgPolicy(): EpsGreedy<O, Int, AS> = egPolicy

    private var targetDQN: IDQN
    override fun getTargetDQN(): IDQN = targetDQN
    override fun setTargetDQN(dqn: IDQN) {
        targetDQN = dqn
    }

    private var lastAction: Int
    private var accuReward : Double
    private var lastMonitor : Int
    private val observableExpRelay = ObservableExpReplay<Int, O>(conf.batchSize, conf.expRepMaxSize)

    init {
        targetDQN = dqn.clone()
        policy = SelectiveDQNPolicy(mdp, dqn)
        egPolicy = EpsGreedy(policy, mdp, conf.updateStart, epsilonNbStep, random, conf.minEpsilon, this)
        lastAction = 0
        accuReward = 0.0
        lastMonitor = -Constants.MONITOR_FREQ
    }

    public override fun postEpoch() {

        if (historyProcessor != null)
            historyProcessor.stopMonitor()

    }

    public override fun preEpoch() {
        lastAction = 0
        accuReward = 0.0

        if (stepCounter - lastMonitor >= Constants.MONITOR_FREQ && historyProcessor != null && getDataManager().isSaveData) {
            lastMonitor = stepCounter
            historyProcessor.startMonitor(getDataManager().videoDir + "/video-" + epochCounter + "-" + stepCounter + ".mp4")
        }
    }

    /**
     * Single step of training
     * @param obs last obs
     *
     * @return relevant info for next step
     */
    override fun trainStep(obs: O): QLearning.QLStepReturn<O> {

        val action: Int?
        val input = getInput(obs)

        historyProcessor?.record(input)
        val skipFrame = historyProcessor?.conf?.skipFrame ?: 1
        val historyLength = historyProcessor?.conf?.historyLength ?: 1
        val updateStart = getConfiguration().updateStart + (getConfiguration().batchSize + historyLength) * skipFrame
        val doSkip = stepCounter % skipFrame != 0

        var maxQ: Double = Double.NaN //ignore if Nan for stats

        //if step of training, just repeat lastAction
        if (doSkip) {
            action = lastAction
        } else {
            val reshapedInput = if (input.shape().size > 2)
                input.reshape(*Learning.makeShape(1, input.shape()))
            else
                input.dup()

            val qs = getCurrentDQN().output(reshapedInput)
            maxQ = mdp.actionSpace.maxQValue(qs)

            action = getEgPolicy().nextAction(reshapedInput)
        }

        lastAction = action!!

        val stepReply = getMdp().step(action)

        accuReward += stepReply.reward * configuration.rewardFactor

        //if it's not a skipped frame, you can do a step of training
        if (!doSkip || stepReply.isDone) {

            val recordTargetObservation = stepReply.observation

            val nhistory = listOf(recordTargetObservation)
            val trans = ObservableTransition(
                    nhistory,
                    action,
                    accuReward,
                    stepReply.isDone,
                    stepReply.observation,
                    mdp.observationSpace
            )
            observableExpRelay.store(trans)

            if (stepCounter > updateStart) {
                val (observations, targets) = getFitTargets(observableExpRelay.getBatch())
                getCurrentDQN().fit(observations, targets)
            }

            accuReward = 0.0
        }


        return QLearning.QLStepReturn(maxQ, getCurrentDQN().latestScore, stepReply)

    }

    private fun getQFilterMatrix(observationHistory: ObservationHistory<O>) : INDArray {
        val qFilterList = observationHistory.map { it.getFilteringActionSpace().computeActionAvailability() }
        return Nd4j.concat(0, *qFilterList.toTypedArray())
    }

    protected fun getFitTargets(transitions: ArrayList<ObservableTransition<Int, O>>): Pair<INDArray, INDArray> {
        if (transitions.size == 0)
            throw IllegalArgumentException("too few transitions")

        val size = transitions.size

        val shape = if (historyProcessor == null) getMdp().observationSpace.shape else historyProcessor.conf.shape
        val nshape = Learning.makeShape(size, shape)

        val obs = Nd4j.create(*nshape)
        val nextObs = Nd4j.create(*nshape)

        val qFilters = Nd4j.create(*nshape)
        val nextQFilters = Nd4j.create(*nshape)

        transitions.forEachIndexed {i, transition ->
            obs.putRow(i, transition.observationHistory.concatinated)
            nextObs.putRow(i, transition.appended.concatinated)
            qFilters.putRow(i, getQFilterMatrix(transition.observationHistory))
            nextQFilters.putRow(i, getQFilterMatrix(transition.appended))
        }

        val dqnOutputAr = qFilters.mul(dqnOutput(obs))
        val dqnOutputNext = nextQFilters.mul(dqnOutput(nextObs))

        var targetDqnOutputNext: INDArray? = null
        var tempQ: INDArray? = null
        var maxActions: INDArray? = null

        if (getConfiguration().isDoubleDQN) {
            targetDqnOutputNext = nextQFilters.mul(targetDqnOutput(nextObs))
            maxActions = Nd4j.argMax(targetDqnOutputNext, 1)
        } else {
            tempQ = Nd4j.max(dqnOutputNext, 1)
        }

        transitions.forEachIndexed {i, transition ->
            val action = transition.action
            var yTar = transition.reward

            if (!transition.isTerminal) {
                val q = if (getConfiguration().isDoubleDQN) {
                    targetDqnOutputNext!!.getDouble(i, maxActions!!.getInt(i))
                } else {
                    tempQ!!.getDouble(i)
                }

                yTar += getConfiguration().gamma * q
            }

            val previousV = dqnOutputAr.getDouble(i, action)
            val lowB = previousV - getConfiguration().errorClamp
            val highB = previousV + getConfiguration().errorClamp
            val clamped = Math.min(highB, Math.max(yTar, lowB))

            dqnOutputAr.putScalar(i, action, clamped)
        }

        return Pair(obs, dqnOutputAr)
    }
}