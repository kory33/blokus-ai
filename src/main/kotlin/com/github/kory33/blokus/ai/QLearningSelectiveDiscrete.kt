package com.github.kory33.blokus.ai

import com.github.kory33.blokus.environment.space.SelectiveDiscreteSpace
import org.deeplearning4j.berkeley.Pair
import org.deeplearning4j.rl4j.learning.Learning
import org.deeplearning4j.rl4j.learning.sync.Transition
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.mdp.MDP
import org.deeplearning4j.rl4j.network.dqn.IDQN
import org.deeplearning4j.rl4j.policy.EpsGreedy
import org.deeplearning4j.rl4j.policy.Policy
import org.deeplearning4j.rl4j.util.Constants
import org.deeplearning4j.rl4j.util.DataManager
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

abstract class QLearningSelectiveDiscrete<O : ISelectiveObservation, AS : SelectiveDiscreteSpace>
        (private val mdp: MDP<O, Int, AS>,
         dqn: IDQN, conf: QLearning.QLConfiguration,
         private val dataManager: DataManager, epsilonNbStep: Int)
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
    private var history: Array<INDArray>?
    private var accuReward : Double
    private var lastMonitor : Int


    init {
        targetDQN = dqn.clone()
        policy = SelectiveDQNPolicy(mdp, dqn)
        egPolicy = EpsGreedy(policy, mdp, conf.updateStart, epsilonNbStep, random, conf.minEpsilon, this)
        lastAction = 0
        history = null
        accuReward = 0.0
        lastMonitor = -Constants.MONITOR_FREQ
    }

    public override fun postEpoch() {

        if (historyProcessor != null)
            historyProcessor.stopMonitor()

    }

    public override fun preEpoch() {
        history = null
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
            if (history == null) {
                historyProcessor?.add(input)
                history = historyProcessor?.history ?: arrayOf(input)
            }

            //concat the history into a single INDArray input
            var hstack = Transition.concat(Transition.dup(history!!))

            //if input is not 2d, you have to append that the batch is 1 length high
            if (hstack.shape().size > 2) {
                hstack = hstack.reshape(*Learning.makeShape(1, hstack.shape()))
            }

            val qs = getCurrentDQN().output(hstack)
            maxQ = mdp.actionSpace.maxQValue(qs)

            action = getEgPolicy().nextAction(hstack)
        }

        lastAction = action!!

        val stepReply = getMdp().step(action)

        accuReward += stepReply.reward * configuration.rewardFactor

        //if it's not a skipped frame, you can do a step of training
        if (!doSkip || stepReply.isDone) {

            val ninput = getInput(stepReply.observation)
            historyProcessor?.add(ninput)

            val nhistory = historyProcessor?.history ?: arrayOf(ninput)

            val trans = Transition(history, action, accuReward, stepReply.isDone, nhistory[0])
            expReplay.store(trans)

            if (stepCounter > updateStart) {
                val targets = setTarget(expReplay.batch)
                getCurrentDQN().fit(targets.first, targets.second)
            }

            history = nhistory
            accuReward = 0.0
        }


        return QLearning.QLStepReturn(maxQ, getCurrentDQN().latestScore, stepReply)

    }


    protected fun setTarget(transitions: ArrayList<Transition<Int>>): Pair<INDArray, INDArray> {
        if (transitions.size == 0)
            throw IllegalArgumentException("too few transitions")

        val size = transitions.size

        val shape = if (historyProcessor == null) getMdp().observationSpace.shape else historyProcessor.conf.shape
        val nshape = Learning.makeShape(size, shape)
        val obs = Nd4j.create(*nshape)
        val nextObs = Nd4j.create(*nshape)
        val actions = IntArray(size)
        val areTerminal = BooleanArray(size)

        for (i in 0..size - 1) {
            val trans = transitions[i]
            areTerminal[i] = trans.isTerminal
            actions[i] = trans.action
            obs.putRow(i, Transition.concat(trans.observation))
            nextObs.putRow(i, Transition.concat(Transition.append(trans.observation, trans.nextObservation)))
        }

        val dqnOutputAr = dqnOutput(obs)

        val dqnOutputNext = dqnOutput(nextObs)
        var targetDqnOutputNext: INDArray? = null

        var tempQ: INDArray? = null
        var getMaxAction: INDArray? = null
        if (getConfiguration().isDoubleDQN) {
            targetDqnOutputNext = targetDqnOutput(nextObs)
            getMaxAction = Nd4j.argMax(targetDqnOutputNext, 1)
        } else {
            tempQ = Nd4j.max(dqnOutputNext, 1)
        }


        for (i in 0..size - 1) {
            var yTar = transitions[i].reward
            if (!areTerminal[i]) {
                var q = 0.0
                if (getConfiguration().isDoubleDQN) {
                    q += targetDqnOutputNext!!.getDouble(i, getMaxAction!!.getInt(i))
                } else
                    q += tempQ!!.getDouble(i)

                yTar += getConfiguration().gamma * q

            }


            val previousV = dqnOutputAr.getDouble(i, actions[i])
            val lowB = previousV - getConfiguration().errorClamp
            val highB = previousV + getConfiguration().errorClamp
            val clamped = Math.min(highB, Math.max(yTar, lowB))

            dqnOutputAr.putScalar(i, actions[i], clamped)
        }

        return Pair(obs, dqnOutputAr)
    }
}