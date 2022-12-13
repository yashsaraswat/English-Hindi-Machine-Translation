# %%
import os
import pandas as pd

# %%


# %%
with open('en-hi/train.en') as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

with open('en-hi/train.hi', "r") as f:
        hindi_text = f.readlines()
        hindi_text = [text.strip("\n") for text in hindi_text]

data = []

for i in range(2000000,4000000):
    data.append(["translate english to hindi", english_text[i], hindi_text[i]])


train_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])
data = []

# %%

# %%
from simpletransformers.t5 import T5Model, T5Args

# %%
import logging
import pandas as pd

# %%
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# %%
model_args = T5Args()
model_args.max_seq_length = 128
model_args.train_batch_size = 32
model_args.eval_batch_size = 32
model_args.num_train_epochs = 5
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.preprocess_inputs = True
model_args.num_return_sequences = 1


# %%

model = T5Model("mt5", "outs",args=model_args)
print(model.config)
# %%
model.train_model(train_df,output_dir='outs2')

# %%



