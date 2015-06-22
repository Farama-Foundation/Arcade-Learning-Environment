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

//import Jama.Matrix;
import java.io.Serializable;

/**
 * Defines a linear regression model. Uses double[] as its feature representation.
 *
 * @author Marc G. Bellemare
 */
public class LinearModel implements Serializable, Cloneable {
    /** Whether we should use the bias weights as well */
    protected boolean useBias = false;
    /** Learning rate for modifying weights */
    protected double alpha = 0.1;

    /** How many features this model expects. */
    protected int numFeatures;
    /** The last prediction made by the model */
    protected double prediction;
    /** The set of weights used to predict */
    protected double[] weights;
    /** The model's bias term */
    protected double bias;

    /**
     * Create a new LinearModel.
     *
     * @param numFeatures The length of the state vector used by this model.
     * @param useBias Whether to use a bias term.
     */
    public LinearModel(int numFeatures, boolean useBias) {
        this.useBias = useBias;
        this.numFeatures = numFeatures;

        // Initialize weights to 0
        weights = new double[numFeatures];
        bias = 0;
        prediction = 0;
    }

    /** Sets the learning rate for this model.
     * 
     * @param alpha
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    /** Returns the learning rate for this model.
     * 
     * @return
     */
    public double getAlpha() {
        return alpha;
    }

    public double[] getWeights() {
        return weights;
    }

    public boolean getUseBias() {
        return useBias;
    }

    public double getPrediction() {
        return prediction;
    }

    /** Makes a prediction for the given feature vector. The prediction is
     *   the dot product of the weight vector with the feature vector.
     * 
     * @param features
     * @return
     */
    public double predict(double[] features) {
        prediction = 0;

        // Dot product
        for (int i = 0; i < features.length; i++) {
            prediction += weights[i] * features[i];
        }

        // Add bias if so desired
        if (useBias)
            prediction += bias;

        return prediction;
    }

    /** Updates the weights by a 'delta' gradient-ish quantity (e.g., TD
     *   error).
     * 
     * @param lastFeatures
     * @param delta
     */
    public void updateWeightsDelta(double[] lastFeatures, double delta) {
        // Update the bias
        if (useBias) {
            bias += alpha * delta;
        }

        // Update other weights
        for (int index = 0; index < lastFeatures.length; index++) {
            double value = lastFeatures[index];

            weights[index] += alpha * (delta * value);
        }
    }
}
