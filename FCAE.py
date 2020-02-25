# necessary keras imports
from keras.models import Model
from keras.layers import Activation, Multiply, Subtract, Add, Lambda, Input, Dense, Concatenate, Dropout
from keras.utils import plot_model
from keras.initializers import RandomUniform, Orthogonal, Zeros, Identity
from keras.optimizers import Nadam
from keras.engine.topology import Layer
import keras.backend as K

# misc
import numpy as np
import numpy.linalg as LA
from numpy import matlib as ML


def activation_function_f(x):
    return 0.5 * (x + np.abs(x))


def activation_function_if(x):
    x1 = 2.0 * (x - 0.5)
    return 2.0 * (x1 + np.power(x1, 3.0) / 3.0 + np.power(x1, 5.0) / 5.0)


class MaskCreateLayerWithModality(Layer):
    def __init__(self, thres, modality, **kwargs):
        super(MaskCreateLayerWithModality, self).__init__(**kwargs)
        self.modality = modality
        self.num_modality = int(np.max(modality))+1
        self.thres = thres

        self.modality_nums = [0] * self.num_modality
        for i in modality:
            self.modality_nums[i] += 1

    def build(self, input_shape):
        super(MaskCreateLayerWithModality, self).build(input_shape)

    def call(self, x, training=None):
        tensorListT = []
        tensorListV = []
        for i in self.modality_nums:
            tensorListT.append( K.repeat_elements( K.cast(K.random_uniform(shape=(1,)) > self.thres,'float32') , i, axis=0 ))
            tensorListV.append( K.repeat_elements( K.cast(K.random_uniform_variable(shape=(1,), low=0, high=1) > self.thres,'float32') , i, axis=0 ))
        maskT = x * K.concatenate(tensorListT)
        maskV = x * K.concatenate(tensorListV)
        return K.in_train_phase(maskT, maskV, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MaskCreateLayerWithModality, self).get_config()
        config['thres']=self.thres
        config['modality']=self.modality
        return config


class FullyConnectedLinearAEHandler:
    def __init__(self, number_features, IR_size=3):
        # Network private attributes
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.IR_layer = None
        self.bias_only_layer = None
        self.encode_layers = []
        self.decode_layers = []

        # Database related informations
        self.modalities = []
        self.number_features = number_features

        # Network parameters
        self.activation_function = 'relu'
        self.layer_width = 10
        self.depth = 7
        self.IR_size = IR_size

        # Optimizable hyperparameters
        self.hyperparamsNames = ["learning_rate",
                                 "batch_size",
                                 "dropout_rate",
                                 "initial_corruption_rate",
                                 "perturb"]
        self.hyperparams = {"learning_rate"          : 0.001,
                            "finetune_bs"            : 256,
                            "dropout_rate"           : 0.5,
                            "initial_corruption_rate": 0.10,
                            "perturb"                : 0.0}
        self.hyperparamsBounds = {"learning_rate"          : [5.0*10e-5, 2.0*10e-2],
                                  "finetune_bs"            : [256, 4096],
                                  "dropout_rate"           : [0.05, 0.5],
                                  "initial_corruption_rate": [0.01, 0.25],
                                  "perturb"                : [0.0, 1.0]}
        self.hyperparamsTypes = {"learning_rate"          : "logistic",
                                 "finetune_bs"            : "logint",
                                 "dropout_rate"           : "logistic",
                                 "initial_corruption_rate": "logistic",
                                 "perturb"                : "interval"}
        self.hyperparamsUsed = {"learning_rate"          : True,
                                "finetune_bs"            : True,
                                "dropout_rate"           : True,
                                "initial_corruption_rate": True,
                                "perturb"                : False}

    # Generate layers for the next based on a size list
    #
    def generate_layers(self):
        self.encode_layers = []
        self.decode_layers = []

        # creating encoding and decoding layers
        prev_size = self.number_features
        enc_inputSize = 0
        for _ in range(self.depth):
            enc_inputSize = enc_inputSize + prev_size
            self.encode_layers.append(Dense(self.layer_width, bias_initializer=RandomUniform(),
                                            kernel_initializer=Orthogonal(),
                                            input_shape=(enc_inputSize,)))
            prev_size = self.layer_width
        self.IR_layer = Dense(self.IR_size, bias_initializer=Zeros(),
                              kernel_initializer=Orthogonal(),
                              input_shape=(enc_inputSize,))
        self.encode_layers.append(self.IR_layer)

        prev_size = self.IR_size
        dec_inputSize = 0
        for _ in range(self.depth):
            dec_inputSize = self.number_features()
            dec_inputSize = dec_inputSize + prev_size
            self.decode_layers.append(Dense(self.layer_width, bias_initializer=Zeros(),
                                            input_shape=(dec_inputSize,)))
            prev_size = self.layer_width
        dec_inputSize = dec_inputSize + prev_size
        self.decode_layers.append( Dense(self.number_features(), input_shape=(dec_inputSize,)))

    # Build the encoder, decoder and autoencoder models according to the layers generated.
    #
    def build_whole_autoencoder(self):
        i = Input(shape=(self.number_features(),), name='value_input')
        m = Input(shape=(self.number_features(),), name='mask_input')

        #create bias only layer for input
        self.bias_only_layer = Dense(units=self.number_features, kernel_initializer=Identity())

        maskLayer = MaskCreateLayerWithModality(self.hyperparams["initial_corruption_rate"],
                                                self.modalities)(m)

        r = [Multiply()([self.bias_only_layer(i), m]),]
        x = [Multiply()([self.bias_only_layer(i), maskLayer]),]
        ri = Input(shape=(self.IR_size,))

        # build encoder and encoder end of the autoencoder
        for encode_layer in self.encode_layers:
            if encode_layer is not self.IR_layer:
                newX = encode_layer(Dropout(self.hyperparams["dropout_rate"])(x[-1]))
                newR = encode_layer(Dropout(self.hyperparams["dropout_rate"])(r[-1]))
            else:
                newX = encode_layer(x[-1])
                newR = encode_layer(r[-1])

            #apply activation and add to input for next layer if we aren't the final part of the encoder
            if encode_layer is not self.IR_layer:
                newX = Activation(self.activation_function)(newX)
                newR = Activation(self.activation_function)(newR)
                x.append(Concatenate()([newX, x[-1]]))
                r.append(Concatenate()([newR, r[-1]]))

            # if we are the final part, don't add activation or concatenation
            else:
                x.append(newX)
                r.append(newR)

        # build decoder end
        #rx = Activation(self.activation_function)(x[-1])
        #rit = Activation(self.activation_function)(ri)

        y = [x[-1],]
        d = [ri,]
        for decode_layer in self.decode_layers:
            if decode_layer is not self.decode_layers[-1]:
                newY = decode_layer(Dropout(self.hyperparams["dropout_rate"])(y[-1]))
                newD = decode_layer(Dropout(self.hyperparams["dropout_rate"])(d[-1]))
            else:
                newY = decode_layer(y[-1])
                newD = decode_layer(d[-1])

            # apply activation and add to input for next layer if we aren't the final part of the decoder
            if decode_layer != self.decode_layers[-1]:
                newY = Activation(self.activation_function)(newY)
                newD = Activation(self.activation_function)(newD)
                y.append(Concatenate()([newY,y[-1]]))
                d.append(Concatenate()([newD,d[-1]]))

            # if we are the final part, don't add activation or concatenation
            else:
                y.append(newY)
                d.append(newD)

        y[-1] = Activation("sigmoid")(y[-1])
        d[-1] = Activation("sigmoid")(d[-1])

        # put on masking layer for faster training (A2 only)
        y.append(Subtract()([i, y[-1]]))
        a1out = Multiply()([y[-1], maskLayer])
        a2out = Lambda(lambda x: 2.0*x)(Multiply()([y[-1], Subtract()([m, maskLayer])]))
        zeros = Add()([a1out, a2out])

        self.encoder = Model(inputs=[i, m], outputs=r[-1])
        self.decoder = Model(inputs=[ri], outputs=d[-1])
        self.autoencoder = Model(inputs=[i, m], outputs=[zeros])

        #pop the trainable kernel off of the bias
        self.bias_only_layer.non_trainable_weights.append(self.bias_only_layer.trainable_weights.pop(0))

    # Primary initialization
    #
    def primary_initialization(self, input_data, input_target, val_data, val_target):
        activation_function_if = self.curator.recommend_activation()[1]

        #get the mean of the data
        numPoints = input_target.shape[0]
        mean = self.decode_layers[-1].get_weights()[1]
        covariance = np.zeros((input_target.shape[1], input_target.shape[1]))
        for col1 in range(mean.shape[0]):
            column1 = input_target[:, col1]
            mean[col1] = np.mean(column1[~np.isnan(column1)].astype(float))
            for col2 in range(col1 + 1, mean.shape[0]):
                column2 = input_target[:, col2]
                mask = np.logical_and(~np.isnan(column1), ~np.isnan(column2))
                covariance[col1, col2] = np.mean((column1[mask] - mean[col1]) * (column2[mask] - mean[col2]))

        # initialize the masking bias with the mean and set kernel to be untrainable
        self.bias_only_layer.set_weights((-mean, np.identity(self.number_features())))

        # get whitened representation
        _, eigen = LA.eigh(covariance)
        curr_input = input_target.copy().astype(float) - ML.repmat(self.bias_only_layer.get_weights()[0], numPoints, 1)
        curr_input[np.isnan(curr_input)] = 0
        output_target = np.matmul(curr_input, eigen)



        #for encoder in self.encode_layers:
        #    numReps = encoder.units
        #    if encoder is self.IR_layer:

        #        curr_input_pad = np.append(np.ones((numPoints, 1)), curr_input, axis=1)
        #        enc, bias = encoder.get_weights()
        #        for r in range(numReps):
        #            target = output_target[:, r]
        #            mask = ~np.isnan(target)
        #            target = target[mask]
        #            input = np.delete(curr_input_pad, np.nonzero(~mask), axis=0)
        #            toInvert = np.matmul(input.T, input)
        #            maxInvertNum = np.max(toInvert)
        #            toInvert = toInvert / maxInvertNum + 0.01 * np.identity(input.shape[1])
        #            B = np.matmul(LA.inv(toInvert), np.matmul(input.T, target) / maxInvertNum)
        #            enc[:, r] = B[1:]

        #            # calculate best intercept given perturbed model (match means for LSTSQs)
        #            bias[r] = np.mean(target) - np.mean(np.matmul(input[:, 1:], enc[:, r]))
        #    else:
        #        enc, bias = encoder.get_weights()
        #    output = self.activation_function_f(-np.matmul(curr_input, enc) + ML.repmat(bias, numPoints, 1))
        #    curr_input = np.append(curr_input, output, axis=1)

        #on the encoding end, use random transformations until
        oldNumInputs = 0
        inputMean = None
        inputCov = None
        for encoder in self.encode_layers:
            numReps = encoder.units
            numInputs = curr_input.shape[1]
            if inputMean is None:
                inputMean = np.zeros((numInputs,))
                inputCov = np.zeros((numInputs, numInputs))
            else:
                newNumInputs = numInputs-oldNumInputs
                inputMean = np.append(inputMean, np.zeros((newNumInputs,)))
                inputCov = np.block([[inputCov, np.zeros((oldNumInputs, newNumInputs))],
                                     [np.zeros((newNumInputs, oldNumInputs)), np.zeros((newNumInputs, newNumInputs))]])
            for col1 in range((oldNumInputs + 1), numInputs):
                col1_data = curr_input[:, col1]
                inputMean[col1] = np.mean(col1_data)
                inputCov[col1, col1] = np.var(col1_data)
            for col1 in range((oldNumInputs + 1), numInputs):
                col1_data = curr_input[:, col1]
                for col2 in range(1, oldNumInputs):
                    col2_data = curr_input[:, col2]
                    inputCov[col1, col2] = np.mean((col1_data - inputMean[col1])*(col2_data - inputMean[col2]))
                for col2 in range(col1+1, numInputs):
                    col2_data = curr_input[:, col2]
                    inputCov[col1, col2] = np.mean((col1_data - inputMean[col1])*(col2_data - inputMean[col2]))
            if encoder is self.IR_layer:
                _, enc = LA.eigh(inputCov)
                enc = enc[:, -numReps:]
                bias = -np.matmul(inputMean, enc)
                encoder.set_weights((enc, bias))
            else:
                (enc,bias) = encoder.get_weights()
            output = activation_function_f(-np.matmul(curr_input, enc) + ML.repmat(bias, numPoints, 1))
            curr_input = np.append(curr_input, output, axis=1)
            oldNumInputs = numInputs

        #on the decoding layers, try capturing as much variance as possible via linear regression
        #to the highest error linear components
        numTotalOutputs = mean.shape[0]
        curr_input = np.append(np.ones((numPoints, 1)), output, axis=1)
        for decoder in self.decode_layers:
            numInputs = curr_input.shape[1]
            numReps = decoder.units
            if numReps == numTotalOutputs:
                output_target = activation_function_if(input_target.copy())
            kernel, bias = decoder.get_weights()

            linResults = []

            #L2 regularized linear regress to capture rth variance (perturb model with this)
            for r in range(numTotalOutputs):
                target = output_target[:, r]
                mask = ~np.isnan(target)
                target = target[mask]
                input = np.delete(curr_input, np.nonzero(~mask), axis=0)
                toInvert = np.matmul(input.T, input)
                maxInvertNum = np.max(toInvert)
                toInvert = toInvert / maxInvertNum + 0.01 * np.identity(numInputs)
                B = np.matmul(LA.inv(toInvert), np.matmul(input.T, target) / maxInvertNum)

                #add some noise and calculate best intercept given perturbed model (match means for LSTSQs)
                if decoder is not self.decode_layers[-1]:
                    linNoised = B[1:] + self.hyperparams["perturb"]*np.random.normal(size=(numInputs-1,))
                else:
                    linNoised = B[1:]
                intNoised = np.mean(target) - np.mean(np.matmul(input[:, 1:], linNoised))

                error = target - np.matmul(input[:,1:], B[1:]) - B[0]
                mse = np.mean(error * error)
                linResults.append((mse, linNoised, intNoised))

            if numReps != numTotalOutputs:
                linResults = sorted(linResults, key=lambda x: -x[0])
            for r in range(numReps):
                kernel[:, r] = linResults[r][1]
                bias[r] = linResults[r][2]

            decoder.set_weights((kernel, bias))
            output = activation_function_f(-np.matmul(curr_input[:, 1:], kernel) + ML.repmat(bias, numPoints, 1))
            curr_input = np.append(curr_input, output, axis=1)

    # Compile the network using a mse loss metric.
    #
    def compile_autoencoder(self):
        optimizer = Nadam(lr=self.hyperparams["learning_rate"])
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
