/*
 * Java Arcade Learning Environment (A.L.E) Agent
 *  Copyright (C) 2011-2012 Marc G. Bellemare <mgbellemare@ualberta.ca>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package ale.rl;

import java.util.ArrayList;

/**
 *  A class that acts accordingly to an epsilon-greedy policy and learns using
 *   SARSA(lambda).
 *
 * @author Marc G. Bellemare
 */
public class SarsaLearner {
    /** The number of actions considered by the learner */
    protected int numActions;

    /** The usual set of SARSA/epsilon-greedy parameters */
    /** Learning rate */
    protected double alpha = 0.1;
    /** Discount factor */
    protected double gamma = 0.999;
    /** Eligibility trace parameter */
    protected double lambda = 0.9;
    /** Probability of a random action */
    protected double epsilon = 0.05;

    /** A set of variables used to perform the SARSA update */
    protected int lastAction;
    protected int action;
    protected double[] lastFeatures;
    protected double[] features;
    
    /** Eligibility traces; need separate traces for each action */
    protected double[][] traces;

    /** Q-value models, one per action */
    protected LinearModel[] valueFunction;

    /** The threshold below which we consider traces to be 0 */
    public static final double minTrace = 0.01;

    /** Creates a new SarsaLearner that expects a feature vector of size
     *   'numFeatures' and takes one of 'numActions' actions.
     * 
     * @param numFeatures
     * @param numActions
     */
    public SarsaLearner(int numFeatures, int numActions) {
        this.numActions = numActions;

        // Create the linear models
        createModels(numActions, numFeatures);
    }

    /** Sets the learning rate for the value function approximators.
     * 
     * @param alpha
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
        for (int a = 0; a < numActions; a++)
            valueFunction[a].setAlpha(alpha);
    }

    /** Begins a new episode with the given feature vector.
     * 
     * @param features
     * @return
     */
    public int agent_start(double[] features) {
        lastFeatures = null;
        traces = null;

        this.features = (double[])features.clone();
        
        return actAndLearn(features, 0.0);
    }

    /** Takes one step in the RL environment. Agent_start must be called at
     *    least once prior to calling agent_step.
     * 
     * @param pReward
     * @param features
     * @return
     */
    public int agent_step(double pReward, double[] features) {
        lastFeatures = this.features;
        this.features = (double[])features.clone();

        return actAndLearn(this.features, pReward);
    }

    /** The very last step, when the agent receives the last reward but not the
     *   subsequent state (because that state is the terminal state).
     * 
     * @param pReward
     */
    public void agent_end(double pReward) {
        learn(features, action, pReward, null, 0);
    }

   /** Take an action and learn from the option given the current state and
     *   current observation.
     *
     * @param features
     * @param pObservation
     * @return
     */
    public int actAndLearn(double[] features, double pReward) {
        lastAction = action;

        // Get the next action
        action = selectAction(features);

        // If this is not the first step...
        if (lastFeatures != null) {
            // Perform a SARSA update
            learn(lastFeatures, lastAction, pReward, features, action);
        }

        return action;
    }

    /** The core of the SARSA learning algorithm. See Sutton and Barto (1998).
     *
     * @param lastFeatures
     * @param lastAction
     * @param reward
     * @param features
     * @param action
     */
    public void learn(double[] lastFeatures, int lastAction,
            double reward, double[] features, int action) {
        // Compute Q(s,a)
        double oldValue = valueFunction[lastAction].predict(lastFeatures);

        // Early exit for diverging agents
        if (Double.isNaN(oldValue) || oldValue >= 10E7)
            throw new RuntimeException("Diverged.");

        // Compute Q(s',a')
        double newValue;

        // ... if s' is null (terminal state), then Q(s',a') is assumed to be 0
        if (features != null)
            newValue = valueFunction[action].predict(features);
        else
            newValue = 0;

        // Compute the TD error
        double delta = reward + gamma * newValue - oldValue;

        // Update the eligibility traces
        updateTraces(lastFeatures, lastAction);

        // With traces, we update *all* models
        for (int a = 0; a < numActions; a++) {
            LinearModel model = valueFunction[a];
            // Perform a TD error linear approximation udpate
            model.updateWeightsDelta(traces[a], delta);
        }
    }

    /** Updates the eligibility traces for all actions. The action that was
     *   actually taken has the current feature vector added to its traces;
     *   all others are simply decayed.
     * 
     * @param features
     * @param lastAction
     */
    public void updateTraces(double[] features, int lastAction) {
        if (traces == null) {
            traces = new double[numActions][];
            traces[lastAction] = (double[])features.clone();

            for (int a = 0; a < numActions; a++)
                if (a != lastAction)
                    traces[a] = new double[features.length];
        }
        else {
            for (int a = 0; a < numActions; a++) {
                // For the selected action, decay its trace and add the new
                //  state vector
                if (a != lastAction)
                    decayTraces(traces[a], gamma*lambda);
                else
                    replaceTraces(traces[a], gamma*lambda, features);
            }
        }
    }

    /** Decays the given eligibility traces.
     * 
     * @param traces
     * @param factor
     */
    protected void decayTraces(double[] traces, double factor) {
        for (int f = 0; f < traces.length; f++)
            traces[f] *= factor;
    }

    /** Replacing traces. This, of course, assumes a sparse feature vector.
     * 
     * @param traces
     * @param factor
     * @param state
     */
    protected void replaceTraces(double[] traces, double factor, double[] state) {
        for (int f = 0; f < traces.length; f++) {
            // If the feature is currently 0, decay its trace
            if (state[f] == 0)
                traces[f] *= factor;
            else
                traces[f] = state[f];
        }
    }

    /** Epsilon-greedy action selection.
     *
     * @param pState
     * @return
     */
    public int selectAction(double[] pState) {
        double[] values = new double[numActions];

        double bestValue = Double.NEGATIVE_INFINITY;
        double worstValue = Double.POSITIVE_INFINITY;

        int bestAction = -1;
        ArrayList<Integer> ties = new ArrayList<Integer>();

        // E-greedy
        if (Math.random() < epsilon) {
            int r = (int)(Math.random() * numActions);
            return r;
        }

        // Greedy selection, with random tie-breaking
        for (int a = 0; a < numActions; a++) {
            double v = valueFunction[a].predict(pState);

            values[a] = v;
            if (v > bestValue) {
                bestValue = v;
                bestAction = a;
                ties.clear();
                ties.add(bestAction);
            }
            else if (v == bestValue) {
                ties.add(a);
            }

            if (v < worstValue)
                worstValue = v;
        }

        // Tie-breaker
        if (ties.size() > 1) {
            int r = (int)(Math.random() * ties.size());
            bestAction = ties.get(r);
        }

        return bestAction;
    }

   /** This method constructs the set of models used by this agent.
     *
     * @param numActions The number of actions available to the agent.
     * @param pObservationDim The dimension of the observation vector.
     */
    protected final void createModels(int numActions, int numFeatures) {
        valueFunction = new LinearModel[numActions];

        for (int a = 0; a < numActions; a++) {
            valueFunction[a] = new LinearModel(numFeatures, true);
            valueFunction[a].setAlpha(alpha);
        }
    }
}
