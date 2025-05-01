## cifar 100 resnet18 with 2048 batch 


```
argparse done Namespace(dataset='cifar100', data_root='./data', backbone='resnet18', time_emb_dim=64, embed_dim=256, epoches=1000)
start
Files already downloaded and verified
Files already downloaded and verified
--- resnet18 on CIFAR100 (100 classes) using cuda ---
dataset config
model config
initialize prototypes
train start
Epoch 001 loss  22.0703 | train 2.5s
Epoch 002 loss  14.7907 | train 2.2s
Epoch 003 loss  11.8199 | train 2.2s
Epoch 004 loss  10.3886 | train 2.2s
Epoch 005 loss   9.5198 | train 2.2s | Acc 13.52% | infer 4.5s
Epoch 006 loss   8.8812 | train 2.2s
Epoch 007 loss   8.3860 | train 2.2s
Epoch 008 loss   7.9125 | train 2.2s
Epoch 009 loss   7.4786 | train 2.2s
Epoch 010 loss   7.1352 | train 2.2s | Acc 21.40% | infer 4.5s
Epoch 011 loss   6.7152 | train 2.2s
Epoch 012 loss   6.3666 | train 2.2s
Epoch 013 loss   6.0491 | train 2.2s
Epoch 014 loss   5.7560 | train 2.3s
Epoch 015 loss   5.4482 | train 2.2s | Acc 28.58% | infer 4.5s
Epoch 016 loss   5.1954 | train 2.2s
Epoch 017 loss   4.9328 | train 2.2s
Epoch 018 loss   4.6470 | train 2.2s
Epoch 019 loss   4.4500 | train 2.2s
Epoch 020 loss   4.2239 | train 2.2s | Acc 34.92% | infer 4.5s
Epoch 021 loss   3.9897 | train 2.2s
Epoch 022 loss   3.7982 | train 2.2s
Epoch 023 loss   3.6260 | train 2.3s
Epoch 024 loss   3.4529 | train 2.2s
Epoch 025 loss   3.3016 | train 2.2s | Acc 39.62% | infer 4.5s
Epoch 026 loss   3.1593 | train 2.2s
Epoch 027 loss   3.0143 | train 2.2s
Epoch 028 loss   2.9184 | train 2.2s
Epoch 029 loss   2.7662 | train 2.2s
Epoch 030 loss   2.6646 | train 2.3s | Acc 42.05% | infer 4.5s
Epoch 031 loss   2.5530 | train 2.3s
Epoch 032 loss   2.4344 | train 2.2s
Epoch 033 loss   2.3318 | train 2.2s
Epoch 034 loss   2.2748 | train 2.2s
Epoch 035 loss   2.1811 | train 2.2s | Acc 44.76% | infer 4.5s
Epoch 036 loss   2.0914 | train 2.2s
Epoch 037 loss   2.0342 | train 2.2s
Epoch 038 loss   1.9479 | train 2.2s
Epoch 039 loss   1.8987 | train 2.3s
Epoch 040 loss   1.8350 | train 2.2s | Acc 45.06% | infer 4.5s
Epoch 041 loss   1.7710 | train 2.2s
Epoch 042 loss   1.6912 | train 2.2s
Epoch 043 loss   1.6183 | train 2.2s
Epoch 044 loss   1.5589 | train 2.2s
Epoch 045 loss   1.5201 | train 2.2s | Acc 46.22% | infer 4.5s
Epoch 046 loss   1.4730 | train 2.3s
Epoch 047 loss   1.4067 | train 2.2s
Epoch 048 loss   1.3692 | train 2.2s
Epoch 049 loss   1.3273 | train 2.2s
Epoch 050 loss   1.2694 | train 2.2s | Acc 46.27% | infer 4.5s
Epoch 051 loss   1.2256 | train 2.2s
Epoch 052 loss   1.1866 | train 2.2s
Epoch 053 loss   1.1385 | train 2.2s
Epoch 054 loss   1.0934 | train 2.2s
Epoch 055 loss   1.0517 | train 2.3s | Acc 47.50% | infer 4.5s
Epoch 056 loss   1.0085 | train 2.2s
Epoch 057 loss   0.9736 | train 2.2s
Epoch 058 loss   0.9401 | train 2.2s
Epoch 059 loss   0.9105 | train 2.2s
Epoch 060 loss   0.8684 | train 2.2s | Acc 46.22% | infer 4.4s
Epoch 061 loss   0.8210 | train 2.2s
Epoch 062 loss   0.7920 | train 2.2s
Epoch 063 loss   0.7631 | train 2.3s
Epoch 064 loss   0.7216 | train 2.3s
Epoch 065 loss   0.7019 | train 2.2s | Acc 46.89% | infer 4.5s
Epoch 066 loss   0.6649 | train 2.2s
Epoch 067 loss   0.6361 | train 2.2s
Epoch 068 loss   0.6101 | train 2.2s
Epoch 069 loss   0.5878 | train 2.2s
Epoch 070 loss   0.5619 | train 2.2s | Acc 46.92% | infer 4.5s
Epoch 071 loss   0.5373 | train 2.2s
Epoch 072 loss   0.5114 | train 2.2s
Epoch 073 loss   0.4849 | train 2.3s
Epoch 074 loss   0.4620 | train 2.3s
Epoch 075 loss   0.4607 | train 2.2s | Acc 47.14% | infer 4.5s
Epoch 076 loss   0.4355 | train 2.3s
Epoch 077 loss   0.4366 | train 2.4s
Epoch 078 loss   0.4041 | train 2.3s
Epoch 079 loss   0.3904 | train 2.3s
Epoch 080 loss   0.3689 | train 2.2s | Acc 47.25% | infer 4.5s
Epoch 081 loss   0.3574 | train 2.3s
Epoch 082 loss   0.3406 | train 2.2s
Epoch 083 loss   0.3350 | train 2.2s
Epoch 084 loss   0.3259 | train 2.2s
Epoch 085 loss   0.3144 | train 2.3s | Acc 46.75% | infer 4.5s
Epoch 086 loss   0.3026 | train 2.3s
Epoch 087 loss   0.2878 | train 2.2s
Epoch 088 loss   0.2713 | train 2.3s
Epoch 089 loss   0.2642 | train 2.3s
Epoch 090 loss   0.2584 | train 2.3s | Acc 47.69% | infer 4.5s
Epoch 091 loss   0.2560 | train 2.2s
Epoch 092 loss   0.2493 | train 2.2s
Epoch 093 loss   0.2407 | train 2.3s
Epoch 094 loss   0.2294 | train 2.3s
Epoch 095 loss   0.2158 | train 2.2s | Acc 46.94% | infer 4.5s
Epoch 096 loss   0.2158 | train 2.3s
Epoch 097 loss   0.2095 | train 2.3s
Epoch 098 loss   0.2130 | train 2.2s
Epoch 099 loss   0.2070 | train 2.3s
Epoch 100 loss   0.2040 | train 2.3s | Acc 47.75% | infer 4.5s
Epoch 101 loss   0.2006 | train 2.2s
Epoch 102 loss   0.1937 | train 2.2s
Epoch 103 loss   0.1792 | train 2.3s
Epoch 104 loss   0.1756 | train 2.3s
Epoch 105 loss   0.1814 | train 2.3s | Acc 46.51% | infer 4.5s
Epoch 106 loss   0.1843 | train 2.3s
Epoch 107 loss   0.1744 | train 2.2s
Epoch 108 loss   0.1739 | train 2.3s
Epoch 109 loss   0.1720 | train 2.2s
Epoch 110 loss   0.1681 | train 2.2s | Acc 46.72% | infer 4.5s
Epoch 111 loss   0.1553 | train 2.3s
Epoch 112 loss   0.1539 | train 2.3s
Epoch 113 loss   0.1531 | train 2.2s
Epoch 114 loss   0.1544 | train 2.3s
Epoch 115 loss   0.1540 | train 2.3s | Acc 47.27% | infer 4.5s
Epoch 116 loss   0.1500 | train 2.2s
Epoch 117 loss   0.1458 | train 2.2s
Epoch 118 loss   0.1527 | train 2.3s
Epoch 119 loss   0.1412 | train 2.2s
Epoch 120 loss   0.1400 | train 2.2s | Acc 48.04% | infer 4.5s
Epoch 121 loss   0.1351 | train 2.2s
Epoch 122 loss   0.1331 | train 2.3s
Epoch 123 loss   0.1357 | train 2.3s
Epoch 124 loss   0.1322 | train 2.4s
Epoch 125 loss   0.1309 | train 2.3s | Acc 46.78% | infer 4.5s
Epoch 126 loss   0.1273 | train 2.2s
Epoch 127 loss   0.1243 | train 2.3s
Epoch 128 loss   0.1239 | train 2.3s
Epoch 129 loss   0.1277 | train 2.3s
Epoch 130 loss   0.1245 | train 2.2s | Acc 46.91% | infer 4.5s
Epoch 131 loss   0.1326 | train 2.3s
Epoch 132 loss   0.1337 | train 2.3s
Epoch 133 loss   0.1288 | train 2.3s
Epoch 134 loss   0.1254 | train 2.3s
Epoch 135 loss   0.1224 | train 2.3s | Acc 46.71% | infer 4.5s
Epoch 136 loss   0.1277 | train 2.3s
Epoch 137 loss   0.1260 | train 2.3s
Epoch 138 loss   0.1208 | train 2.3s
Epoch 139 loss   0.1173 | train 2.3s
Epoch 140 loss   0.1126 | train 2.2s | Acc 46.75% | infer 4.5s
Epoch 141 loss   0.1133 | train 2.3s
Epoch 142 loss   0.1140 | train 2.3s
Epoch 143 loss   0.1084 | train 2.3s
Epoch 144 loss   0.1127 | train 2.3s
Epoch 145 loss   0.1102 | train 2.3s | Acc 47.16% | infer 4.5s
Epoch 146 loss   0.1076 | train 2.3s
Epoch 147 loss   0.1090 | train 2.3s
Epoch 148 loss   0.1009 | train 2.3s
Epoch 149 loss   0.1016 | train 2.2s
Epoch 150 loss   0.1086 | train 2.2s | Acc 46.63% | infer 4.5s
Epoch 151 loss   0.1024 | train 2.3s
Epoch 152 loss   0.1080 | train 2.3s
Epoch 153 loss   0.0981 | train 2.3s
Epoch 154 loss   0.0988 | train 2.3s
Epoch 155 loss   0.0981 | train 2.3s | Acc 47.36% | infer 4.5s
Epoch 156 loss   0.0937 | train 2.2s
Epoch 157 loss   0.0984 | train 2.2s
Epoch 158 loss   0.1018 | train 2.2s
Epoch 159 loss   0.1068 | train 2.2s
Epoch 160 loss   0.1041 | train 2.4s | Acc 47.00% | infer 4.5s
Epoch 161 loss   0.1015 | train 2.4s
Epoch 162 loss   0.0990 | train 2.3s
Epoch 163 loss   0.1004 | train 2.3s
Epoch 164 loss   0.0902 | train 2.3s
Epoch 165 loss   0.0991 | train 2.2s | Acc 46.99% | infer 4.5s
Epoch 166 loss   0.0931 | train 2.3s
Epoch 167 loss   0.0914 | train 2.2s
Epoch 168 loss   0.0908 | train 2.2s
Epoch 169 loss   0.0906 | train 2.3s
Epoch 170 loss   0.0884 | train 2.3s | Acc 46.94% | infer 4.5s
Epoch 171 loss   0.0913 | train 2.3s
Epoch 172 loss   0.0948 | train 2.3s
Epoch 173 loss   0.0901 | train 2.4s
Epoch 174 loss   0.0879 | train 2.4s
Epoch 175 loss   0.0867 | train 2.4s | Acc 47.46% | infer 4.6s
Epoch 176 loss   0.0897 | train 2.3s
Epoch 177 loss   0.0869 | train 2.3s
Epoch 178 loss   0.0856 | train 2.3s
Epoch 179 loss   0.0880 | train 2.3s
Epoch 180 loss   0.0839 | train 2.2s | Acc 46.57% | infer 4.5s
Epoch 181 loss   0.0823 | train 2.3s
Epoch 182 loss   0.0847 | train 2.2s
Epoch 183 loss   0.0856 | train 2.3s
Epoch 184 loss   0.0862 | train 2.3s
Epoch 185 loss   0.0857 | train 2.3s | Acc 46.96% | infer 4.5s
Epoch 186 loss   0.0868 | train 2.3s
Epoch 187 loss   0.0810 | train 2.3s
Epoch 188 loss   0.0805 | train 2.3s
Epoch 189 loss   0.0844 | train 2.3s
Epoch 190 loss   0.0828 | train 2.4s | Acc 46.76% | infer 4.6s
Epoch 191 loss   0.0829 | train 2.3s
Epoch 192 loss   0.0897 | train 2.2s
Epoch 193 loss   0.0890 | train 2.3s
Epoch 194 loss   0.0837 | train 2.2s
Epoch 195 loss   0.0824 | train 2.3s | Acc 46.57% | infer 4.5s
Epoch 196 loss   0.0855 | train 2.4s
Epoch 197 loss   0.0781 | train 2.3s
Epoch 198 loss   0.0738 | train 2.3s
Epoch 199 loss   0.0709 | train 2.4s
Epoch 200 loss   0.0709 | train 2.3s | Acc 47.13% | infer 4.5s
Epoch 201 loss   0.0671 | train 2.3s
Epoch 202 loss   0.0761 | train 2.3s
Epoch 203 loss   0.0720 | train 2.3s
Epoch 204 loss   0.0735 | train 2.3s
Epoch 205 loss   0.0769 | train 2.3s | Acc 46.75% | infer 4.5s
Epoch 206 loss   0.0751 | train 2.3s
Epoch 207 loss   0.0751 | train 2.4s
Epoch 208 loss   0.0718 | train 2.4s
Epoch 209 loss   0.0688 | train 2.2s
Epoch 210 loss   0.0733 | train 2.2s | Acc 46.48% | infer 4.5s
Epoch 211 loss   0.0792 | train 2.2s
Epoch 212 loss   0.0773 | train 2.3s
Epoch 213 loss   0.0802 | train 2.2s
Epoch 214 loss   0.0729 | train 2.2s
Epoch 215 loss   0.0758 | train 2.2s | Acc 47.32% | infer 4.5s
Epoch 216 loss   0.0652 | train 2.3s
Epoch 217 loss   0.0710 | train 2.3s
Epoch 218 loss   0.0718 | train 2.3s
Epoch 219 loss   0.0674 | train 2.2s
Epoch 220 loss   0.0632 | train 2.2s | Acc 47.41% | infer 4.5s
Epoch 221 loss   0.0629 | train 2.3s
Epoch 222 loss   0.0625 | train 2.3s
Epoch 223 loss   0.0670 | train 2.2s
Epoch 224 loss   0.0708 | train 2.3s
Epoch 225 loss   0.0711 | train 2.3s | Acc 46.91% | infer 4.5s
```
