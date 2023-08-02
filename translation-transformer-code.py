
# Affiche les détails du GPU NVIDIA pour s'assurer qu'il est disponible.
!nvidia-smi

# Installe la bibliothèque tensorflow_datasets, qui offre un ensemble de jeux de données prêts à être utilisés avec TensorFlow.
!pip install tensorflow_datasets

# Met à jour et installe la bibliothèque tensorflow-text, utile pour certaines opérations de traitement du texte avec TensorFlow.
!pip install -U tensorflow-text

# Importe divers modules Python standard et bibliothèques nécessaires.
import collections      # Fournit des structures de données alternatives comme les dictionnaires, les ensembles et les listes.
import logging          # Facilite la génération de journaux pour suivre le déroulement de l'exécution.
import os               # Permet d'interagir avec le système d'exploitation, par exemple pour gérer les chemins de fichiers.
import pathlib          # Offre une interface orientée objet pour gérer les systèmes de fichiers et les chemins.
import re               # Fournit des fonctions pour travailler avec des expressions régulières.
import string           # Contient des opérations courantes sur les chaînes.
import sys              # Fournit un accès à certaines variables et fonctions utilisées par l'interpréteur Python.
import time             # Fournit des fonctions liées au temps, comme la pause et la mesure du temps d'exécution.

# Importe les bibliothèques pour la manipulation de données et la visualisation.
import numpy as np      # Bibliothèque pour le calcul scientifique avec Python.
import matplotlib.pyplot as plt  # Bibliothèque pour la création de visualisations et de graphiques.

# Importe des bibliothèques spécifiques à TensorFlow.
import tensorflow_datasets as tfds  # Fournit un accès aux jeux de données.
import tensorflow_text as text      # Bibliothèque pour le traitement du texte avec TensorFlow.
import tensorflow as tf             # Importe TensorFlow lui-même.

# Affiche le nom du GPU utilisé par TensorFlow, s'il est disponible.
print(tf.test.gpu_device_name())

# Configure le module de journalisation de TensorFlow pour supprimer les avertissements.
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Charger le jeu de données 'pt_to_en' (portugais à anglais) de TED HRLR Translate
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)

# Séparer les données en données d'entraînement et de validation
train_examples, val_examples = examples['train'], examples['validation']

# Afficher quelques exemples de phrases en portugais et en anglais
for pt_examples, en_examples in train_examples.batch(3).take(1):
  for pt in pt_examples.numpy():
    print(pt.decode('utf-8'))

  print()

  for en in en_examples.numpy():
    print(en.decode('utf-8'))

# Nom du modèle pour la tokenisation
model_name = "ted_hrlr_translate_pt_en_converter"

# Télécharge le modèle de tokenisation
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)

# Charge le modèle de tokenisation
tokenizers = tf.saved_model.load(model_name)

# Liste les méthodes disponibles pour le tokenizer en anglais
[item for item in dir(tokenizers.en) if not item.startswith('_')]

# Imprime des exemples encodés
for en in en_examples.numpy():
  print(en.decode('utf-8'))

# Tokenise les exemples
encoded = tokenizers.en.tokenize(en_examples)

# Imprime les exemples tokenisés
for row in encoded.to_list():
  print(row)

# Détokenise les exemples pour revenir au texte original
round_trip = tokenizers.en.detokenize(encoded)
for line in round_trip.numpy():
  print(line.decode('utf-8'))

# Regarde les tokens utilisés pour l'encodage
tokens = tokenizers.en.lookup(encoded)
tokens

# Fonction pour tokeniser les paires de phrases (pt et en)
def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convertit de ragged à dense, avec remplissage de zéros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convertit de ragged à dense, avec remplissage de zéros.
    en = en.to_tensor()
    return pt, en

# Paramètres pour le traitement par lots et le mélange
BUFFER_SIZE = 20000
BATCH_SIZE = 64

# Crée des lots pour le jeu de données
def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))

# Crée les lots d'entraînement et de validation
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

# Fonction pour calculer les angles de l'encodage positionnel
def get_angles(pos, i, d_model):
  # Calcule les taux d'angle pour l'encodage
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  # Renvoie la multiplication des positions et des taux d'angle
  return pos * angle_rates

# Fonction pour générer l'encodage positionnel
def positional_encoding(position, d_model):
  # Calcul des angles radiaux pour l'encodage positionnel
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  # Applique la fonction sinus aux colonnes à indices pairs
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # Applique la fonction cosinus aux colonnes à indices impairs
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  # Ajoute une nouvelle dimension à angle_rads
  pos_encoding = angle_rads[np.newaxis, ...]

  # Retourne l'encodage positionnel sous forme de tensor
  return tf.cast(pos_encoding, dtype=tf.float32)

# Création et affichage de l'encodage positionnel
n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
# Reshape et transpose pour la visualisation
pos_encoding = pos_encoding[0]
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))
# Visualisation de l'encodage positionnel
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Profondeur')
plt.xlabel('Position')
plt.colorbar()
plt.show()

# Visualisation séparée des fonctions sinus et cosinus
plt.figure(figsize=(12, 6))

# Sélectionnez une dimension spécifique à visualiser
dim_to_visualize = 50
# Si vous souhaitez visualiser une autre dimension, changez simplement la valeur de dim_to_visualize

plt.plot(pos_encoding[2 * dim_to_visualize, :], label='Sinus Encoding')
plt.plot(pos_encoding[2 * dim_to_visualize + 1, :], label='Cosinus Encoding')
plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title(f'Encodage positionnel pour la dimension {dim_to_visualize}')
plt.legend()
plt.grid(True)
plt.show()

# Fonction pour créer un masque de padding
def create_padding_mask(seq):
  # Identifie les zéros dans la séquence
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # Retourne le masque pour la séquence
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# Exemple de création de masque de padding
x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)

# Fonction pour créer un masque look-ahead
def create_look_ahead_mask(size):
  # Crée un masque triangulaire supérieur
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

# Exemple de création de masque look-ahead
x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
temp

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calcule les poids d'attention.

    Arguments:
        q: forme de la requête == (..., seq_len_q, depth)
        k: forme de la clé == (..., seq_len_k, depth)
        v: forme de la valeur == (..., seq_len_v, depth_v)
        mask: tensor de flottants avec forme adaptable
              à (..., seq_len_q, seq_len_k). Par défaut à None.

    Retour:
        output, attention_weights
    """
    # Calcule le produit scalaire de q et k
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # Échelle matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # Si un masque est fourni, l'ajoute
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # Calcule les poids d'attention
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # Obtenir le tensor de sortie en multipliant les poids et v
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Fonction pour afficher les résultats d'attention
def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Les poids d’attention sont :')
    print(temp_attn)
    print('Le résultat est :')
    print(temp_out)

# Exemples d'utilisation
np.set_printoptions(suppress=True)
temp_k = tf.constant([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype=tf.float32)
temp_v = tf.constant([[1, 0], [10, 0], [100, 5], [1000, 6]], dtype=tf.float32)
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)
print_out(temp_q, temp_k, temp_v)

# Quelques exemples de requêtes.
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

temp_q = tf.constant([[0, 0, 10],
                      [0, 10, 0],
                      [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # Nombre de têtes pour l'attention multi-tête
        self.d_model = d_model  # Dimension du modèle

        # Assure que la dimension du modèle est divisible par le nombre de têtes
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # Calcul de la profondeur de chaque tête

        # Couches denses pour les requêtes, les clés et les valeurs
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)  # Couche dense finale

    def split_heads(self, x, batch_size):
        """Divise la dernière dimension en (num_heads, depth).
        Transpose le résultat pour obtenir la forme (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Passage des requêtes, clés et valeurs à travers les couches denses
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Divise les requêtes, clés et valeurs pour l'attention multi-tête
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Calcule l'attention échelonnée
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Fusion des têtes d'attention
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # Passe l'attention concaténée à travers une dernière couche dense
        output = self.dense(concat_attention)

        return output, attention_weights

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))
out, attn = temp_mha(y, k=y, q=y, mask=None)
out.shape, attn.shape

def point_wise_feed_forward_network(d_model, dff):
    # Crée un réseau feed-forward pour le modèle point par point
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # Première couche dense
        tf.keras.layers.Dense(d_model)  # Deuxième couche dense
    ])

sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        # Couches pour l'attention multi-tête et le réseau feed-forward point par point
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Normalisations des couches et des "dropout"
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # Attention multi-tête
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Réseau feed-forward point par point
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

sample_encoder_layer = EncoderLayer(512, 8, 2048)
sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
sample_encoder_layer_output.shape

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        # Couches pour l'attention multi-tête (x2) et le réseau feed-forward point par point
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Normalisations des couches et des "dropout"
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        # Première attention multi-tête
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Deuxième bloc d'attention multi-têtes avec les sorties de l'encodeur comme clés et valeurs.
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # Normalisation et addition résiduelle.

        # Réseau feed-forward point par point.
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Normalisation et addition résiduelle.

        # La sortie est la représentation transformée avec les poids d'attention de deux blocs.
        return out3, attn_weights_block1, attn_weights_block2

# Création et test d'un exemple de couche décodeur.
sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, False, None, None)
print(sample_decoder_layer_output.shape)  # Affiche la forme de la sortie : (batch_size, target_seq_len, d_model)

# Classe pour l'encodeur
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Initialisation de l'embedding des mots et de l'encoding des positions
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        # Création des couches d'encodeurs
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Ajout de l'embedding et de l'encoding des positions
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        # Passer x à travers les couches d'encodeurs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # Retourne la sortie de l'encodeur

# Exemple de l'encodeur
sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500, maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
print(sample_encoder_output.shape)  # Affiche la forme de la sortie

# Classe pour le décodeur
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Initialisation de l'embedding des mots et de l'encoding des positions
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        # Création des couches de décodeurs
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Ajout de l'embedding et de l'encoding des positions
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        # Passer x à travers les couches de décodeurs
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights  # Retourne la sortie du décodeur et les poids d'attention

# Exemple de décodeur
sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000, maximum_position_encoding=5000)
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
output, attn = sample_decoder(temp_input, enc_output=sample_encoder_output, training=False, look_ahead_mask=None, padding_mask=None)
output.shape, attn['decoder_layer2_block2'].shape

# Classe pour le modèle complet du transformateur
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

# Exemple du transformateur
sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500, target_vocab_size=8000, pe_input=10000, pe_target=6000)
temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
fn_out, _ = sample_transformer([temp_input, temp_target], training=False)
fn_out.shape

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

# Définition d'une classe pour un planning d'apprentissage personnalisé
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    # Initialisation du modèle dimensionnel
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    # Initialisation des étapes d'échauffement
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    # Convertir le tenseur step en flottant (float32)
    step = tf.cast(step, tf.float32)

    # Calcul des arguments pour la planification de l'apprentissage
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    # Retourne la valeur du taux d'apprentissage pour l'étape donnée
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Instanciation du taux d'apprentissage personnalisé
learning_rate = CustomSchedule(d_model)

# Initialisation de l'optimiseur Adam avec le taux d'apprentissage personnalisé
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Visualisation de la courbe de taux d'apprentissage
temp_learning_rate_schedule = CustomSchedule(d_model)
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Taux d'apprentissage")
plt.xlabel("Étape d'entrainement")

# Objet pour calculer la perte
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# Fonction de perte personnalisée
def loss_function(real, pred):
  # Création d'un masque pour les séquences non nulles
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  # Application du masque à la perte
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  # Retour de la perte moyenne
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Fonction pour calculer la précision
def accuracy_function(real, pred):
  # Comparaison des valeurs réelles avec les prédictions
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  # Création d'un masque pour les séquences non nulles
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  # Application du masque aux précisions
  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  # Retour de la précision moyenne
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# Mesures pour suivre la perte et la précision durant l'entraînement
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# Création de l'objet Transformer avec des configurations spécifiques
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(), # Taille du vocabulaire portugais
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(), # Taille du vocabulaire anglais
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

# Chemin pour enregistrer les checkpoints durant l'entraînement
checkpoint_path = "./checkpoints/train"

# Configuration du checkpoint
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

# Gestionnaire de checkpoints
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Si un checkpoint existe déjà, on restaure le plus récent
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Dernier checkpoint restauré!!')

EPOCHS = 20

# L'annotation @tf.function permet d'accélérer l'exécution en traçant la fonction sous forme de graphe TensorFlow
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  # Préparation des cibles
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  # Calcul des prédictions et de la perte
  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp], training=True)
    loss = loss_function(tar_real, predictions)

  # Calcul des gradients et mise à jour des poids
  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

# Listes pour stocker les valeurs de perte et de précision
losses = []
accuracies = []

# Boucle d'entraînement
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # Boucle sur les batches d'entraînement
  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

    losses.append(train_loss.result().numpy())
    accuracies.append(train_accuracy.result().numpy())

    # Affichage de la perte et de la précision tous les 50 batches
    if batch % 50 == 0:
      print(f'Époque {epoch + 1} Batch {batch} Perte {train_loss.result():.4f} Précision {train_accuracy.result():.4f}')

  # Sauvegarde du checkpoint tous les 5 époques
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Checkpoint sauvegardé pour l\'époque {epoch+1} à {ckpt_save_path}')

  print(f'Époque {epoch + 1} Perte {train_loss.result():.4f} Précision {train_accuracy.result():.4f}')
  print(f'Temps pris pour 1 époque: {time.time() - start:.2f} secondes\n')

# Définition de la classe Translator pour effectuer la traduction
class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=20):
    # Traitement de la phrase d'entrée
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]
    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
    encoder_input = sentence

    # Initialisation des tokens de début et de fin pour l'anglais
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    # Boucle pour générer la traduction
    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)
      predictions = predictions[:, -1:, :]
      predicted_id = tf.argmax(predictions, axis=-1)

      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    # Récupération et détokenisation du texte traduit
    output = tf.transpose(output_array.stack())
    text = tokenizers.en.detokenize(output)[0]
    tokens = tokenizers.en.lookup(output)[0]

    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)
    return text, tokens, attention_weights

translator = Translator(tokenizers, transformer)

# Fonction pour afficher les résultats de traduction
def print_translation(sentence, tokens, ground_truth):
  print(f'{"Entrée":15s}: {sentence}')
  print(f'{"Prédiction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Vérité terrain":15s}: {ground_truth}')

# Exemples de traductions
sentence = "este é um problema que temos que resolver."
ground_truth = "c'est un problème que nous devons résoudre."
translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

sentence = "os meus vizinhos ouviram sobre esta ideia."
ground_truth = "and my neighboring homes heard about this idea ."

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

sentence = "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram."
ground_truth = "so i \'ll just share with you some stories very quickly of some magical things that have happened ."

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

# Définir une phrase en portugais et sa traduction réelle en anglais
sentence = "este é o primeiro livro que eu fiz."
ground_truth = "this is the first book i've ever done."

# Utilisez le traducteur pour obtenir la traduction prédite et les poids d'attention
translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

def plot_attention_head(in_tokens, translated_tokens, attention):
  """Trace les poids d'attention pour une tête spécifique."""

  # Le modèle n'a pas généré `<START>` dans la sortie. On le saute.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)

# Sélectionnez une tête d'attention spécifique pour la visualisation
head = 0
attention_heads = tf.squeeze(
  attention_weights['decoder_layer4_block2'], 0)
attention = attention_heads[head]

# Convertir la phrase d'entrée en tokens
in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.pt.lookup(in_tokens)[0]

# Tracer les poids d'attention pour la tête sélectionnée
plot_attention_head(in_tokens, translated_tokens, attention)

def plot_attention_weights(sentence, translated_tokens, attention_heads):
  """Trace les poids d'attention pour toutes les têtes."""

  in_tokens = tf.convert_to_tensor([sentence])
  in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
  in_tokens = tokenizers.pt.lookup(in_tokens)[0]

  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)
    plot_attention_head(in_tokens, translated_tokens, head)
    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()

plot_attention_weights(sentence, translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])

# Testez avec une autre phrase
sentence = "Eu li sobre triceratops na enciclopédia."
ground_truth = "I read about triceratops in the encyclopedia."

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

plot_attention_weights(sentence, translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()

print(*losses, sep=", ")

print(*accuracies, sep=", ")