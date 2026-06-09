import os
import base64
from IPython.display import HTML

def show_gif(gif_path):
    with open(gif_path, "rb") as f:
        data_uri = base64.b64encode(f.read()).decode("utf-8")
    return HTML(f'<img src="data:image/gif;base64,{data_uri}" style="margin:5px;" />')

def ovf_it(idx):
    return show_gif(f'/kaggle/input/overfitarc/ovf_{idx}.gif')


ovf_it('16de56c4_train_1')


ovf_it('53fb4810_train_0')


ovf_it('53fb4810_train_1')


ovf_it('db0c5428_train_0')


ovf_it('db0c5428_train_2')


ovf_it('142ca369_train_0')


ovf_it('142ca369_train_1')


ovf_it('142ca369_train_2')


ovf_it('4c3d4a41_train_0')


ovf_it('4c3d4a41_train_1')


ovf_it('e376de54_train_0')


ovf_it('e376de54_train_1')


ovf_it('e376de54_train_2')


ovf_it('409aa875_train_0')


ovf_it('409aa875_train_1')


ovf_it('409aa875_train_2')


ovf_it('35ab12c3_train_0')


ovf_it('35ab12c3_train_1')


ovf_it('35ab12c3_train_2')


ovf_it('e3721c99_train_0')


ovf_it('e3721c99_train_1')


ovf_it('135a2760_train_0')


ovf_it('135a2760_train_1')


ovf_it('13e47133_train_0')


ovf_it('13e47133_train_1')


ovf_it('13e47133_train_2')


ovf_it('16b78196_train_0')


ovf_it('16b78196_train_1')


ovf_it('195c6913_train_0')


ovf_it('195c6913_train_1')


ovf_it('195c6913_train_2')


ovf_it('d8e07eb2_train_0')


ovf_it('d8e07eb2_train_1')


ovf_it('d8e07eb2_train_2')


ovf_it('d8e07eb2_train_3')


ovf_it('d8e07eb2_train_4')


ovf_it('7ed72f31_train_0')


ovf_it('7ed72f31_train_1')


ovf_it('28a6681f_train_0')


ovf_it('28a6681f_train_1')


ovf_it('28a6681f_train_2')


ovf_it('da515329_train_0')


ovf_it('da515329_train_1')


ovf_it('da515329_train_2')


ovf_it('dbff022c_train_0')


ovf_it('dbff022c_train_1')


ovf_it('dbff022c_train_2')


ovf_it('de809cff_train_0')


ovf_it('de809cff_train_1')

