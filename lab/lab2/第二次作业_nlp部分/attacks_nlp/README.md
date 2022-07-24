
## 项目结构

|
|-advtrain                          对抗训练
    |-fgsm.py                       fsgm
    |-pgd.py                        pgd
|-attacks                           攻击方式
    |-text_bugger.py                textbugguer攻击
    |-transformation_rules.py       自定义策略攻击         
|-datas                             训练数据
|-models                            模型，TextCNN

### python
|
|- run_train.py  正常训练
|- run_adv_train.py  对抗训练
|- run_batch_attacks.py 黑白盒攻击

