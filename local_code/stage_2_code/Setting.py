'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting

class Setting(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load_data()

        # run MethodModule
        self.method.data = {'train': loaded_data['train'], 'test': loaded_data['test']}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

        