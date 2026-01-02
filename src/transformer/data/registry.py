from transformer.data.builders.seq2seq import *
from transformer.data.builders.lm import *


SEQ2SEQ_BUILDERS = {
    "ted_talks": load_ted_talks
}

LM_BUILDERS = {
    "tiny_shakespeare": load_tiny_shakespeare
}