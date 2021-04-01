import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from transformers import TFDistilBertModel


def distilbert_model(input_shape, transformer_model,
                     output_shape=96, output_activation='softmax',
                     optimizer='Adam', optimizer_params={'lr': 1e-5},
                     loss='categorical_crossentropy', metrics=None):

    input_ids = Input((input_shape,), dtype=tf.int32)
    input_mask = Input((input_shape,), dtype=tf.int32)

    transformer_encoder = TFDistilBertModel.from_pretrained(transformer_model, from_pt=True,
                                                            output_hidden_states=True)
    outputs = transformer_encoder.distilbert(input_ids, attention_mask=input_mask)

    x = outputs[0]
    x = GlobalAveragePooling1D()(x)
    output = Dense(output_shape, activation=output_activation)(x)

    model = Model(inputs=[input_ids, input_mask], outputs=output)
    model.compile(loss=loss, metrics=metrics,
                  optimizer=getattr(optimizers, optimizer)(**optimizer_params))

    return model
