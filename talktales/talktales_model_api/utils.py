import tensorflow as tf
import keras
import numpy as np
import json
import requests
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PredictModel():
    frame_length = 256
    frame_step = 160
    fft_length = 384
    batch_size = 32
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )
    tf_serving_url = "https://tf-serving-eresbajlya-uc.a.run.app"
    tfidf_vectorizer = TfidfVectorizer()

    def scoring_prediction_and_label(self, prediction, label):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([label, prediction])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]


    def decode_batch_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # bisa pakai beam search juga --> untuk complex tasks
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False, beam_width=100, top_paths=1)[0][0]
        # iterate over the results and get back the text
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(self.num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    def CTCLoss(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def preprocess_audio_file(self, wav_bytes):
        wav_tensor = tf.convert_to_tensor(wav_bytes, dtype=tf.string)
        audio, _ = tf.audio.decode_wav(wav_tensor)
        audio = tf.squeeze(audio, axis=-1)

        #   # 2. Get the mel spectrogram
        stft = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length)
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(spectrogram, tf.signal.linear_to_mel_weight_matrix(num_mel_bins=193, num_spectrogram_bins=spectrogram.shape[-1], sample_rate=16000), 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

        #   # 3. Normalization
        means = tf.math.reduce_mean(log_mel_spectrogram, axis=1, keepdims=True)
        stddevs = tf.math.reduce_std(log_mel_spectrogram, axis=1, keepdims=True)
        normalized_mel_spectrogram = (log_mel_spectrogram - means) / (stddevs + 1e-10)
        return normalized_mel_spectrogram

    def find_difference_word(self, target, prediction):
        incorrect_words = []
        target_words = target.lower().split()
        predicted_words = prediction.lower().split()
        for word in target_words:
            if word not in predicted_words:
                possibilities = difflib.get_close_matches(word, predicted_words, n=3, cutoff=0.65)
                if len(possibilities)>0:
                    pass
                else:
                    incorrect_words.append(word)
        return incorrect_words
    
    def markup_difference_in_html(self, string, difference_list):
        string_list = string.split()
        string_lower = string.lower().split()
        
        for word in difference_list:
            index = string_lower.index(word)
            escaped_word = f'<font color=#DA0000>{string_list[index]}</font>'
            string_list[index] = escaped_word
        return " ".join(string_list)
    
    def predict(self, wav_bytes, target_label):
        preprocessed = self.preprocess_audio_file(wav_bytes)
        preprocessed = tf.expand_dims(preprocessed, axis=0)
        spectrogram = preprocessed.numpy().tolist()
        json_data = json.dumps({"instances": spectrogram})
        response = requests.post(f'{self.tf_serving_url}/v1/models/model3_saved_model:predict', data=json_data)
        response = json.loads(response.text)

        if 'predictions' in response:
            decoded = self.decode_batch_prediction(np.array(response['predictions']))[0]
            label = target_label.lower()
            score = self.scoring_prediction_and_label(decoded, label)
            word_diff = self.find_difference_word(label, decoded)
            markup_html = self.markup_difference_in_html(target_label, word_diff)

            return {
                "prediction": decoded,
                "scores": score,
                "feedback": "Incorrect" if score < 0.8 else "Correct",
                "target": label,
                "per_word_eval": "Incorrect" if len(word_diff) > 0 else "Correct",
                "missing_word": word_diff,
                "markup_html": markup_html
            }
        else:
            return "Error when processing model"

