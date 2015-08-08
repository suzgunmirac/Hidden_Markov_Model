using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HiddenMarkovModelTrain
{
    class logHMM
    {
        public int numStates; // number of states

        public int numObservations; // size of the observations

        public double[] prior; // initial states (prior)

        public double[,] A; // transitionMatrix

        public double[,] B; // emissionMatrix


        public double[,] fwd; // forward-vars
        public double[,] bwd; // backward-vars

        public double log_epsilon = Math.Log (double.Epsilon);

        public double log_prob_forward;

        public int time_duration;

        /** Constructor
         * @ param numStates
         * @ param numObservations
         **/
        public logHMM(int numStates, int numObservations)
        {
            this.numStates = numStates;
            this.numObservations = numObservations;

            prior = new double[numStates];

            A = new double[numStates, numStates];
            B = new double[numStates, numObservations];

            Random rand_generator = new Random();

            for (int i = 0; i < numStates; i++)
            {
                
                for (int j = 0; j < numStates; j++)
                {
                   A[i, j] = Math.Log(1.0 / numStates); // uniform distribution for the transition matrix : A
                    // A[i, j] = Math.Log((rand_generator.NextDouble()));
                }

                for (int j = 0; j < numObservations; j++)
                {
                    B[i, j] = Math.Log(1.0 / numObservations); // uniform distribution for the emission matrix: B
                    // B[i, j] = Math.Log((rand_generator.NextDouble()));
                }

                 prior[i] = Math.Log(1.0 / numStates); // unifor distribution for the prior (initial state probabilities): pi  
            }

        }


        /** Baum-Welch Training 
         * @param data : the training set
         **/
        public void baumWelchTrain(int[] data)
        {
            time_duration = data.Length;

            double [,] gamma = new double [numStates, time_duration];

            double[,] alpha_num = new double[numStates, numStates]; // new A (transition matrix)
            double [] alpha_denom = new double [numStates];

            double[,] beta_num = new double[numStates, numObservations]; // new B (emission matrix)
            double [,] beta_denom = new double [numStates, numObservations];

            double[] a0  = new double[numStates]; // new prior 


            //calculate the foward and backward variables on the model

            fwd = forwardCalc(data);
            bwd = backwardCalc(data);

            /** RE-ESTIMATION and UPDATE **/
            
            // re-estimation of the prior=a0 (initial state) probabilities and UPDATE
            
            for (int i = 0; i < numStates; i++)
            {
                a0 [i] = totalNumberOfTransCalc (0, i);
    
                /**
                for (int j = 0; j<numStates; j++){
                    double gamma_t_j = fwd[0, j] + bwd [0, j];
                    a0_denom [i] = log_Sum (a0_denom[i], gamma_t_j);
                    Console.WriteLine("a0_denom: " + a0_denom[i].ToString());
                }
                 **/

                // Update prior with a0:
                // which means prior = a0;
                prior[i] = a0[i];
                Console.WriteLine("prior: " + prior[i].ToString());

            }
            

            // re-estimation of the the transtition prob
            for (int t = 0; t<time_duration -2; t++){

                for (int i=0; i<numStates; i++){
                    alpha_denom[i] = double.NaN;

                    for (int j = 0; j<numStates; j++){
                        alpha_num[i, j] = double.NaN;
                        

                        double xi = fwd[t, i] + A[i, j] + B[j, data[t + 1]] + bwd[t + 1, j] - totalNumberOfTransCalc (t, i);
                        alpha_num [i,j] = log_Sum (alpha_num [i,j], xi);
                        alpha_denom[i] = log_Sum (alpha_denom[i], xi);
                        // Console.WriteLine("alpha_num[i,j]: " + alpha_denom[i].ToString());
                    }
                }
            }

            // update A with alpha: A = alpha

            for (int i = 0; i<numStates; i++){
                
                for (int j = 0; j<numStates; j++){

                    A [i, j] = alpha_num[i,j] - alpha_denom[i];
                }
            }

            // re-estimaton of emission prob 

            for (int j = 0; j<numStates; j++){

                for (int k = 0; k<numObservations; k++){
                    beta_num[j, k] = double.NaN;
                    beta_denom[j, k] = double.NaN;

                    for (int t = 0; t < time_duration; t++)
                    {
                        double gamma_t_j = fwd[t, j] + bwd[t, j] - totalNumberOfTransCalc (t, j);

                        // if data[t] is equal to k, then multiply gamma_t_j by 1, otherwise 0

                        if (data[t] == k ){
                            beta_num[j,k] = log_Sum (beta_num[j,k], gamma_t_j);
                        }
                        beta_denom[j, k] = log_Sum(beta_denom[j, k], gamma_t_j);

                    }

                    // UPDATE B [j,k] with beta [j,k]
                    B[j, k] = beta_num[j, k] - beta_denom[j, k];
                }
            }

        }


        /** Algorithm for Forward-Variable(s) Calculatiın at the state i, time t
         * @param data -> integer olarak geliyor
         * @return an array, fwd, that contains the forward-variables
         * multi-dimens olduguna dikkat et!!! 
         **/
        public double[,] forwardCalc(int[] data)
        {
            int time_duration = data.Length;

            fwd = new double[time_duration, numStates];

            for (int t = 0; t < time_duration; t++)
            {
                for (int i = 0; i < numStates; i++)
                {
                    fwd[t, i] = double.NaN;
                }
            }


            // at time t = 0
            // initialization of the initial prob states
            for (int i = 0; i < numStates; i++)
            {
                fwd[0, i] = prior[i] + B[i, data[0]];
            }


            // induction step
            for (int t = 0; t < time_duration - 1; t++) // 0 <= t <= T-2
            {
                for (int j = 0; j < numStates; j++)
                {
                    for (int i = 0; i < numStates; i++)
                    {
                        double prob = fwd[t, i] + A[i, j] + B[j, data[t + 1]];
                        fwd[t + 1, j] = log_Sum(fwd[t + 1, j], prob);
                    }
                }
            }

           log_prob_forward = double.NaN;

            for (int i = 0; i < numStates; i++)
            {
                log_prob_forward = log_Sum(log_prob_forward, fwd[time_duration - 1, i]);
            }

            Console.WriteLine("log_prob_forward: " + log_prob_forward.ToString());

            return fwd;
        }

        /** Algorith m for the Backward-Variables(s) Calculation at the state i, time t
         * @ param data -> int alıyor
         * @ return an array, bwd, that contains the backward-variables
         * multi-dimension olduguna dikkat et
         **/
        public double[,] backwardCalc(int[] data)
        {
            int time_duration = data.Length;

            bwd = new double[time_duration, numStates];

           
            for (int t = 0; t < time_duration; t++)
            {
                for (int i = 0; i < numStates; i++)
                {
                    bwd[t, i] = double.NaN;
                }
            }
            

            // at time t = 0 -> daha doğrusu t = t (amma velakin beta oldugundan t = 0)
            // initialization 
            for (int i = 0; i < numStates; i++)
            {
                bwd[time_duration - 1, i] = 0; // log (1) = 0; -> normalde 1 olacakti ancak simdi 0 olacak log'dan dolayi
            }

            // induction step
            for (int t = time_duration - 2; t >= 0; t--)
            {
                for (int i = 0; i < numStates; i++)
                {
                    for (int j = 0; j < numStates; j++)
                    {
                        double prob = A[i, j] + B[j, data[t + 1]] + bwd[t + 1, j];
                        bwd[t, i] = log_Sum(bwd[t,i], prob);

                        /**
                        if (bwd[t, i] = 0) // For debugging purposes
                            Console.WriteLine("t and i: " + t + "," + i + " -> " + bwd[t,i].ToString());
                         **/
                    }
                }
            }

            return bwd;
        }

        double log_Sum (double p1, double p2)
        {
            if (double.IsNaN(p1) || double.IsInfinity(p1))
            {
                return p2;
            }
            if (double.IsNaN(p2) || double.IsInfinity(p2))
            {
                return p1;

            }
            if (p1 > p2)
            {
                return p1 + Math.Log(Math.Exp(p2 - p1));
            }
            else
            {
                return p2 + Math.Log(Math.Exp(p1 - p2));
            }
        }

        double totalNumberOfTransCalc (int t, int i)
        {
            double total_trans_num = fwd[t, i] + bwd[t, i];
            double total_trans_denom = double.NaN;

            for (int j = 0; j < numStates; j++)
            {
               // Console.WriteLine("fwd[t,j] and bwd [t,j]: " + fwd [t, j] + " and " + bwd[t,j]);

                total_trans_denom = log_Sum(total_trans_denom, (fwd[t, j] + bwd[t, j]));
            }

            return total_trans_num - total_trans_denom;
        }

    }
}
