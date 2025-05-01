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

## cifar100 with resnet-50 
```
argparse done Namespace(dataset='cifar100', data_root='./data', backbone='resnet50', time_emb_dim=64, embed_dim=256, epoches=1000)
start
Files already downloaded and verified
Files already downloaded and verified
--- resnet50 on CIFAR100 (100 classes) using cuda ---
dataset config
model config
initialize prototypes
train start
Epoch 001 loss 4042.3892 | train 5.7s
Epoch 002 loss 3674.3424 | train 5.2s
Epoch 003 loss 3392.2352 | train 5.2s
Epoch 004 loss 3271.5288 | train 5.2s
Epoch 005 loss 3191.4748 | train 5.2s | Acc 1.00% | infer 18.6s
Epoch 006 loss 3134.8787 | train 5.2s
Epoch 007 loss 3091.0608 | train 5.2s
Epoch 008 loss 3054.5206 | train 5.2s
Epoch 009 loss 3018.9900 | train 5.2s
Epoch 010 loss 2983.6493 | train 5.2s | Acc 1.00% | infer 18.7s
Epoch 011 loss 2949.9456 | train 5.2s
Epoch 012 loss 2925.8568 | train 5.2s
Epoch 013 loss 2905.3613 | train 5.1s
Epoch 014 loss 2881.3117 | train 5.2s
Epoch 015 loss 2858.4690 | train 5.1s | Acc 1.00% | infer 18.7s
Epoch 016 loss 2841.7191 | train 5.2s
Epoch 017 loss 2821.4125 | train 5.2s
Epoch 018 loss 2800.1196 | train 5.2s
Epoch 019 loss 2786.6836 | train 5.2s
Epoch 020 loss 2767.7454 | train 5.2s | Acc 1.00% | infer 18.6s
Epoch 021 loss 2746.0399 | train 5.2s
Epoch 022 loss 2729.2357 | train 5.2s
Epoch 023 loss 2712.6105 | train 5.2s
Epoch 024 loss 2694.0798 | train 5.2s
Epoch 025 loss 2676.6162 | train 5.2s | Acc 1.01% | infer 18.7s
Epoch 026 loss 2662.3342 | train 5.2s
Epoch 027 loss 2646.9069 | train 5.2s
Epoch 028 loss 2629.1158 | train 5.2s
Epoch 029 loss 2611.3839 | train 5.2s
Epoch 030 loss 2595.8706 | train 5.2s | Acc 1.51% | infer 18.7s
Epoch 031 loss 2577.7049 | train 5.2s
Epoch 032 loss 2560.1215 | train 5.2s
Epoch 033 loss 2544.1152 | train 5.2s
Epoch 034 loss 2527.1126 | train 5.2s
Epoch 035 loss 2512.0004 | train 5.2s | Acc 1.07% | infer 18.7s
Epoch 036 loss 2493.3987 | train 5.2s
Epoch 037 loss 2476.6592 | train 5.2s
Epoch 038 loss 2461.0605 | train 5.2s
Epoch 039 loss 2445.1204 | train 5.2s
Epoch 040 loss 2428.2273 | train 5.2s | Acc 1.12% | infer 18.6s
Epoch 041 loss 2411.2619 | train 5.2s
Epoch 042 loss 2395.3017 | train 5.2s
Epoch 043 loss 2378.8641 | train 5.2s
Epoch 044 loss 2363.4103 | train 5.1s
Epoch 045 loss 2346.8922 | train 5.2s | Acc 1.23% | infer 18.7s
Epoch 046 loss 2331.5893 | train 5.2s
Epoch 047 loss 2327.9892 | train 5.3s
Epoch 048 loss 2322.7177 | train 5.2s
Epoch 049 loss 2293.5539 | train 5.2s
Epoch 050 loss 2281.5245 | train 5.2s | Acc 1.06% | infer 18.7s
Epoch 051 loss 2271.5490 | train 5.2s
Epoch 052 loss 2249.4328 | train 5.2s
Epoch 053 loss 2234.0466 | train 5.2s
Epoch 054 loss 2219.2440 | train 5.2s
Epoch 055 loss 2203.7084 | train 5.2s | Acc 1.01% | infer 18.6s
Epoch 056 loss 2191.5644 | train 5.2s
Epoch 057 loss 2176.8342 | train 5.2s
Epoch 058 loss 2179.6886 | train 5.2s
Epoch 059 loss 2148.8514 | train 5.2s
Epoch 060 loss 2128.0790 | train 5.2s | Acc 0.87% | infer 18.6s
Epoch 061 loss 2111.2912 | train 5.2s
Epoch 062 loss 2101.6225 | train 5.2s
Epoch 063 loss 2082.1406 | train 5.3s
Epoch 064 loss 2070.6437 | train 5.2s
Epoch 065 loss 2060.0881 | train 5.2s | Acc 1.04% | infer 18.6s
Epoch 066 loss 2037.3300 | train 5.2s
Epoch 067 loss 2022.1367 | train 5.2s
Epoch 068 loss 2009.1075 | train 5.2s
Epoch 069 loss 2000.2284 | train 5.2s
Epoch 070 loss 2012.6263 | train 5.2s | Acc 0.89% | infer 18.7s
Epoch 071 loss 1983.3816 | train 5.2s
Epoch 072 loss 1965.0442 | train 5.3s
Epoch 073 loss 1952.1046 | train 5.2s
Epoch 074 loss 1933.9629 | train 5.2s
Epoch 075 loss 1915.3444 | train 5.3s | Acc 1.00% | infer 18.7s
Epoch 076 loss 1899.6594 | train 5.3s
Epoch 077 loss 1885.8729 | train 5.2s
Epoch 078 loss 1875.8882 | train 5.2s
Epoch 079 loss 1862.9994 | train 5.2s
Epoch 080 loss 1855.2600 | train 5.2s | Acc 0.86% | infer 18.6s
Epoch 081 loss 1841.1363 | train 5.2s
Epoch 082 loss 1822.4255 | train 5.3s
Epoch 083 loss 1808.5358 | train 5.2s
Epoch 084 loss 1794.6865 | train 5.2s
Epoch 085 loss 1780.3399 | train 5.2s | Acc 1.00% | infer 18.7s
Epoch 086 loss 1767.5888 | train 5.2s
Epoch 087 loss 1755.0759 | train 5.2s
Epoch 088 loss 1740.8762 | train 5.2s
Epoch 089 loss 1727.4587 | train 5.2s
Epoch 090 loss 1716.9742 | train 5.2s | Acc 1.01% | infer 18.6s
Epoch 091 loss 1703.1737 | train 5.2s
Epoch 092 loss 1689.5515 | train 5.3s
Epoch 093 loss 1677.1526 | train 5.2s
Epoch 094 loss 1665.4655 | train 5.3s
Epoch 095 loss 1652.6404 | train 5.2s | Acc 0.98% | infer 18.6s
Epoch 096 loss 1639.7869 | train 5.2s
Epoch 097 loss 1628.3626 | train 5.2s
Epoch 098 loss 1616.1264 | train 5.2s
Epoch 099 loss 1604.0308 | train 5.2s
Epoch 100 loss 1591.7651 | train 5.2s | Acc 1.00% | infer 18.6s
Epoch 101 loss 1579.7211 | train 5.2s
Epoch 102 loss 1567.7321 | train 5.2s
Epoch 103 loss 1556.6895 | train 5.2s
Epoch 104 loss 1545.0515 | train 5.2s
Epoch 105 loss 1532.5916 | train 5.3s | Acc 1.00% | infer 18.7s
Epoch 106 loss 1521.3779 | train 5.3s
Epoch 107 loss 1510.4563 | train 5.3s
Epoch 108 loss 1498.2647 | train 5.3s
Epoch 109 loss 1487.4896 | train 5.2s
Epoch 110 loss 1476.1105 | train 5.2s | Acc 1.00% | infer 18.7s
Epoch 111 loss 1464.9319 | train 5.2s
Epoch 112 loss 1453.8437 | train 5.2s
Epoch 113 loss 1442.6013 | train 5.2s
Epoch 114 loss 1431.9737 | train 5.2s
Epoch 115 loss 1420.6227 | train 5.2s | Acc 1.00% | infer 18.7s
Epoch 116 loss 1409.8054 | train 5.2s
Epoch 117 loss 1399.8020 | train 5.2s
Epoch 118 loss 1388.4143 | train 5.4s
Epoch 119 loss 1376.9558 | train 5.3s
Epoch 120 loss 1367.1666 | train 5.3s | Acc 1.01% | infer 18.7s
Epoch 121 loss 1356.4174 | train 5.3s
Epoch 122 loss 1346.3400 | train 5.2s
Epoch 123 loss 1335.1656 | train 5.3s
Epoch 124 loss 1325.5512 | train 5.2s
Epoch 125 loss 1314.4485 | train 5.2s | Acc 0.99% | infer 18.7s
Epoch 126 loss 1304.7562 | train 5.4s
Epoch 127 loss 1294.3746 | train 5.2s
Epoch 128 loss 1284.2761 | train 5.2s
Epoch 129 loss 1274.9468 | train 5.2s
Epoch 130 loss 1264.4488 | train 5.3s | Acc 1.00% | infer 18.6s
Epoch 131 loss 1254.3817 | train 5.3s
Epoch 132 loss 1245.0059 | train 5.4s
Epoch 133 loss 1234.7231 | train 5.4s
Epoch 134 loss 1224.8266 | train 5.3s
Epoch 135 loss 1215.4212 | train 5.3s | Acc 1.08% | infer 18.8s
Epoch 136 loss 1206.3493 | train 5.3s
Epoch 137 loss 1196.3412 | train 5.3s
Epoch 138 loss 1186.5459 | train 5.2s
Epoch 139 loss 1177.4737 | train 5.2s
Epoch 140 loss 1168.3425 | train 5.3s | Acc 1.04% | infer 18.7s
Epoch 141 loss 1158.5251 | train 5.3s
Epoch 142 loss 1149.3024 | train 5.3s
Epoch 143 loss 1140.3267 | train 5.3s
Epoch 144 loss 1131.5616 | train 5.3s
Epoch 145 loss 1122.1125 | train 5.3s | Acc 1.23% | infer 18.7s
Epoch 146 loss 1113.2870 | train 5.3s
Epoch 147 loss 1104.1702 | train 5.3s
Epoch 148 loss 1094.9756 | train 5.3s
Epoch 149 loss 1087.3150 | train 5.3s
Epoch 150 loss 1077.6202 | train 5.4s | Acc 1.01% | infer 18.7s
Epoch 151 loss 1069.1116 | train 5.2s
Epoch 152 loss 1060.3986 | train 5.3s
Epoch 153 loss 1051.3423 | train 5.3s
Epoch 154 loss 1043.3793 | train 5.3s
Epoch 155 loss 1034.8638 | train 5.3s | Acc 1.66% | infer 18.7s
Epoch 156 loss 1026.0823 | train 5.3s
Epoch 157 loss 1018.4669 | train 5.3s
Epoch 158 loss 1009.8522 | train 5.3s
Epoch 159 loss 1001.7335 | train 5.3s
Epoch 160 loss 993.2609 | train 5.3s | Acc 1.28% | infer 18.8s
Epoch 161 loss 985.7376 | train 5.3s
Epoch 162 loss 977.2656 | train 5.3s
Epoch 163 loss 969.3245 | train 5.2s
Epoch 164 loss 961.1606 | train 5.4s
Epoch 165 loss 952.5919 | train 5.2s | Acc 1.41% | infer 18.7s
Epoch 166 loss 945.2708 | train 5.2s
Epoch 167 loss 937.2034 | train 5.2s
Epoch 168 loss 929.4618 | train 5.2s
Epoch 169 loss 922.0332 | train 5.2s
Epoch 170 loss 913.8894 | train 5.2s | Acc 1.23% | infer 18.6s
Epoch 171 loss 906.8508 | train 5.2s
Epoch 172 loss 899.4479 | train 5.3s
Epoch 173 loss 891.3047 | train 5.2s
Epoch 174 loss 884.5943 | train 5.2s
Epoch 175 loss 876.9270 | train 5.2s | Acc 2.51% | infer 18.6s
Epoch 176 loss 869.6613 | train 5.3s
Epoch 177 loss 862.4989 | train 5.2s
Epoch 178 loss 855.0381 | train 5.3s
Epoch 179 loss 847.6412 | train 5.2s
Epoch 180 loss 840.2141 | train 5.2s | Acc 2.45% | infer 18.6s
Epoch 181 loss 833.9610 | train 5.2s
Epoch 182 loss 826.5747 | train 5.2s
Epoch 183 loss 819.6343 | train 5.2s
Epoch 184 loss 812.6997 | train 5.2s
Epoch 185 loss 805.6999 | train 5.3s | Acc 3.15% | infer 18.8s
Epoch 186 loss 799.2143 | train 5.2s
Epoch 187 loss 792.5397 | train 5.2s
Epoch 188 loss 785.2654 | train 5.1s
Epoch 189 loss 778.8115 | train 5.2s
Epoch 190 loss 772.4876 | train 5.2s | Acc 4.29% | infer 18.6s
Epoch 191 loss 765.5979 | train 5.3s
Epoch 192 loss 758.9327 | train 5.2s
Epoch 193 loss 752.6664 | train 5.2s
Epoch 194 loss 746.1759 | train 5.2s
Epoch 195 loss 739.5988 | train 5.2s | Acc 3.08% | infer 18.7s
Epoch 196 loss 733.2029 | train 5.2s
Epoch 197 loss 727.1423 | train 5.3s
Epoch 198 loss 720.7266 | train 5.2s
Epoch 199 loss 714.4584 | train 5.2s
Epoch 200 loss 708.5661 | train 5.1s | Acc 3.54% | infer 18.7s
Epoch 201 loss 702.1822 | train 5.3s
Epoch 202 loss 696.2217 | train 5.2s
Epoch 203 loss 690.1273 | train 5.2s
Epoch 204 loss 684.4025 | train 5.2s
Epoch 205 loss 678.3488 | train 5.2s | Acc 4.22% | infer 18.6s
Epoch 206 loss 672.7462 | train 5.2s
Epoch 207 loss 666.6035 | train 5.2s
Epoch 208 loss 660.6390 | train 5.3s
Epoch 209 loss 654.9099 | train 5.2s
Epoch 210 loss 649.0793 | train 5.1s | Acc 4.21% | infer 18.7s
Epoch 211 loss 643.5355 | train 5.2s
Epoch 212 loss 638.2339 | train 5.2s
Epoch 213 loss 632.0723 | train 5.2s
Epoch 214 loss 626.9753 | train 5.2s
Epoch 215 loss 621.2401 | train 5.2s | Acc 4.05% | infer 18.6s
Epoch 216 loss 615.4952 | train 5.2s
Epoch 217 loss 610.5256 | train 5.2s
Epoch 218 loss 604.9797 | train 5.2s
Epoch 219 loss 599.3169 | train 5.2s
Epoch 220 loss 594.6256 | train 5.2s | Acc 5.69% | infer 18.7s
Epoch 221 loss 589.0546 | train 5.2s
Epoch 222 loss 583.9980 | train 5.3s
Epoch 223 loss 578.7778 | train 5.2s
Epoch 224 loss 573.5145 | train 5.2s
Epoch 225 loss 568.1004 | train 5.2s | Acc 6.57% | infer 18.6s
Epoch 226 loss 563.3377 | train 5.3s
Epoch 227 loss 558.2590 | train 5.2s
Epoch 228 loss 553.1814 | train 5.2s
Epoch 229 loss 548.4506 | train 5.2s
Epoch 230 loss 543.7514 | train 5.2s | Acc 5.65% | infer 18.7s
Epoch 231 loss 538.8743 | train 5.2s
Epoch 232 loss 533.7920 | train 5.2s
Epoch 233 loss 528.8582 | train 5.2s
Epoch 234 loss 524.1221 | train 5.2s
Epoch 235 loss 519.4669 | train 5.2s | Acc 7.03% | infer 18.7s
Epoch 236 loss 514.7272 | train 5.2s
Epoch 237 loss 509.9254 | train 5.2s
Epoch 238 loss 505.6811 | train 5.2s
Epoch 239 loss 500.9191 | train 5.2s
Epoch 240 loss 496.3298 | train 5.2s | Acc 7.79% | infer 18.7s
Epoch 241 loss 492.0834 | train 5.2s
Epoch 242 loss 487.2258 | train 5.3s
Epoch 243 loss 482.8891 | train 5.3s
Epoch 244 loss 478.7299 | train 5.2s
Epoch 245 loss 474.4758 | train 5.3s | Acc 8.36% | infer 18.7s
Epoch 246 loss 470.0525 | train 5.2s
Epoch 247 loss 465.7696 | train 5.3s
Epoch 248 loss 461.2428 | train 5.2s
Epoch 249 loss 457.2106 | train 5.3s
Epoch 250 loss 452.9991 | train 5.3s | Acc 9.09% | infer 18.7s
Epoch 251 loss 448.5766 | train 5.2s
Epoch 252 loss 444.4904 | train 5.2s
Epoch 253 loss 440.8244 | train 5.2s
Epoch 254 loss 436.3742 | train 5.2s
Epoch 255 loss 432.3077 | train 5.2s | Acc 8.19% | infer 18.6s
Epoch 256 loss 428.5292 | train 5.2s
Epoch 257 loss 424.6768 | train 5.2s
Epoch 258 loss 420.6195 | train 5.2s
Epoch 259 loss 416.8698 | train 5.3s
Epoch 260 loss 412.3923 | train 5.2s | Acc 8.94% | infer 18.6s
Epoch 261 loss 408.9435 | train 5.2s
Epoch 262 loss 405.6103 | train 5.2s
Epoch 263 loss 401.3278 | train 5.3s
Epoch 264 loss 397.7480 | train 5.2s
Epoch 265 loss 393.9377 | train 5.2s | Acc 9.91% | infer 18.7s
Epoch 266 loss 390.4724 | train 5.3s
Epoch 267 loss 386.6193 | train 5.2s
Epoch 268 loss 383.0419 | train 5.2s
Epoch 269 loss 379.5996 | train 5.3s
Epoch 270 loss 376.3296 | train 5.2s | Acc 10.19% | infer 18.7s
Epoch 271 loss 372.5471 | train 5.2s
Epoch 272 loss 368.9026 | train 5.2s
Epoch 273 loss 365.6006 | train 5.2s
Epoch 274 loss 361.9431 | train 5.2s
Epoch 275 loss 358.6799 | train 5.3s | Acc 11.13% | infer 18.6s
Epoch 276 loss 355.0656 | train 5.2s
Epoch 277 loss 352.0935 | train 5.2s
Epoch 278 loss 348.7817 | train 5.2s
Epoch 279 loss 345.2263 | train 5.2s
Epoch 280 loss 342.1737 | train 5.3s | Acc 10.93% | infer 18.6s
Epoch 281 loss 338.8477 | train 5.2s
Epoch 282 loss 335.5312 | train 5.2s
Epoch 283 loss 332.4087 | train 5.2s
Epoch 284 loss 329.1590 | train 5.2s
Epoch 285 loss 326.2396 | train 5.2s | Acc 11.37% | infer 18.7s
Epoch 286 loss 322.7233 | train 5.2s
Epoch 287 loss 320.0339 | train 5.2s
Epoch 288 loss 316.9869 | train 5.2s
Epoch 289 loss 313.8331 | train 5.2s
Epoch 290 loss 310.7410 | train 5.2s | Acc 11.03% | infer 18.7s
Epoch 291 loss 307.9323 | train 5.3s
Epoch 292 loss 304.9013 | train 5.2s
Epoch 293 loss 301.9162 | train 5.2s
Epoch 294 loss 299.1851 | train 5.2s
Epoch 295 loss 295.9500 | train 5.2s | Acc 11.83% | infer 18.6s
Epoch 296 loss 293.2836 | train 5.2s
Epoch 297 loss 290.5421 | train 5.3s
Epoch 298 loss 287.8235 | train 5.2s
Epoch 299 loss 285.0038 | train 5.2s
Epoch 300 loss 282.1137 | train 5.2s | Acc 13.02% | infer 18.7s
Epoch 301 loss 279.3167 | train 5.3s
Epoch 302 loss 276.7539 | train 5.2s
Epoch 303 loss 274.0206 | train 5.2s
Epoch 304 loss 271.5207 | train 5.3s
Epoch 305 loss 268.7959 | train 5.2s | Acc 13.17% | infer 18.6s
Epoch 306 loss 267.1030 | train 5.2s
Epoch 307 loss 264.0339 | train 5.2s
Epoch 308 loss 261.4169 | train 5.2s
Epoch 309 loss 258.8004 | train 5.2s
Epoch 310 loss 256.0682 | train 5.1s | Acc 11.10% | infer 18.7s
Epoch 311 loss 253.5355 | train 5.2s
Epoch 312 loss 251.0877 | train 5.2s
Epoch 313 loss 248.5368 | train 5.3s
Epoch 314 loss 246.3679 | train 5.2s
Epoch 315 loss 243.4787 | train 5.2s | Acc 12.94% | infer 18.7s
Epoch 316 loss 241.0770 | train 5.2s
Epoch 317 loss 238.7218 | train 5.2s
Epoch 318 loss 236.3951 | train 5.2s
Epoch 319 loss 234.2214 | train 5.2s
Epoch 320 loss 231.7721 | train 5.3s | Acc 13.18% | infer 18.7s
Epoch 321 loss 229.5080 | train 5.2s
Epoch 322 loss 227.1874 | train 5.3s
Epoch 323 loss 225.1483 | train 5.2s
Epoch 324 loss 222.7672 | train 5.2s
Epoch 325 loss 220.5616 | train 5.2s | Acc 13.77% | infer 18.7s
Epoch 326 loss 218.4512 | train 5.2s
Epoch 327 loss 216.0638 | train 5.2s
Epoch 328 loss 214.0595 | train 5.3s
Epoch 329 loss 211.6884 | train 5.1s
Epoch 330 loss 209.9078 | train 5.2s | Acc 13.94% | infer 18.6s
Epoch 331 loss 207.8727 | train 5.2s
Epoch 332 loss 205.5718 | train 5.2s
Epoch 333 loss 203.7038 | train 5.2s
Epoch 334 loss 201.5825 | train 5.2s
Epoch 335 loss 199.4485 | train 5.2s | Acc 15.91% | infer 18.7s
Epoch 336 loss 197.6251 | train 5.2s
Epoch 337 loss 195.5152 | train 5.3s
Epoch 338 loss 193.6136 | train 5.2s
Epoch 339 loss 191.6318 | train 5.3s
Epoch 340 loss 189.4944 | train 5.2s | Acc 16.90% | infer 18.7s
Epoch 341 loss 187.8050 | train 5.2s
Epoch 342 loss 185.8980 | train 5.2s
Epoch 343 loss 184.0189 | train 5.2s
Epoch 344 loss 182.3808 | train 5.2s
Epoch 345 loss 180.4058 | train 5.2s | Acc 16.96% | infer 18.7s
Epoch 346 loss 178.2900 | train 5.2s
Epoch 347 loss 176.6286 | train 5.2s
Epoch 348 loss 174.7144 | train 5.2s
Epoch 349 loss 173.1203 | train 5.2s
Epoch 350 loss 171.3835 | train 5.3s | Acc 15.81% | infer 18.6s
Epoch 351 loss 169.3991 | train 5.2s
Epoch 352 loss 167.8020 | train 5.2s
Epoch 353 loss 165.9764 | train 5.2s
Epoch 354 loss 164.4499 | train 5.2s
Epoch 355 loss 162.7100 | train 5.3s | Acc 16.43% | infer 18.7s
Epoch 356 loss 160.9916 | train 5.2s
Epoch 357 loss 159.3455 | train 5.3s
Epoch 358 loss 157.7331 | train 5.3s
Epoch 359 loss 156.2442 | train 5.2s
Epoch 360 loss 154.5512 | train 5.2s | Acc 16.36% | infer 18.6s
Epoch 361 loss 152.9408 | train 5.1s
Epoch 362 loss 151.4545 | train 5.2s
Epoch 363 loss 149.7501 | train 5.2s
Epoch 364 loss 148.2374 | train 5.2s
Epoch 365 loss 146.5562 | train 5.2s | Acc 17.92% | infer 18.7s
Epoch 366 loss 145.2796 | train 5.2s
Epoch 367 loss 143.6142 | train 5.2s
Epoch 368 loss 142.1568 | train 5.2s
Epoch 369 loss 140.7188 | train 5.2s
Epoch 370 loss 139.4204 | train 5.2s | Acc 16.94% | infer 18.7s
Epoch 371 loss 137.7611 | train 5.2s
Epoch 372 loss 136.3109 | train 5.2s
Epoch 373 loss 134.8417 | train 5.2s
Epoch 374 loss 133.5563 | train 5.2s
Epoch 375 loss 132.0766 | train 5.2s | Acc 17.46% | infer 18.7s
Epoch 376 loss 130.7245 | train 5.2s
Epoch 377 loss 129.4691 | train 5.2s
Epoch 378 loss 128.0934 | train 5.2s
Epoch 379 loss 126.5249 | train 5.2s
Epoch 380 loss 125.4141 | train 5.2s | Acc 18.08% | infer 18.7s
Epoch 381 loss 123.9930 | train 5.3s
Epoch 382 loss 122.7344 | train 5.2s
Epoch 383 loss 121.4176 | train 5.2s
Epoch 384 loss 120.0879 | train 5.2s
Epoch 385 loss 118.8361 | train 5.2s | Acc 18.75% | infer 18.7s
Epoch 386 loss 117.5916 | train 5.3s
Epoch 387 loss 116.3541 | train 5.2s
Epoch 388 loss 115.0743 | train 5.2s
Epoch 389 loss 113.8602 | train 5.2s
Epoch 390 loss 112.6485 | train 5.2s | Acc 19.25% | infer 18.7s
Epoch 391 loss 111.3415 | train 5.3s
Epoch 392 loss 110.1834 | train 5.2s
Epoch 393 loss 109.0189 | train 5.2s
Epoch 394 loss 107.9991 | train 5.3s
Epoch 395 loss 106.7325 | train 5.3s | Acc 19.62% | infer 18.7s
Epoch 396 loss 105.4515 | train 5.3s
Epoch 397 loss 104.4960 | train 5.2s
Epoch 398 loss 103.4256 | train 5.2s
Epoch 399 loss 102.2099 | train 5.2s
Epoch 400 loss 101.2873 | train 5.2s | Acc 18.71% | infer 18.6s
Epoch 401 loss 100.1165 | train 5.2s
Epoch 402 loss  99.0136 | train 5.2s
Epoch 403 loss  97.9413 | train 5.2s
Epoch 404 loss  96.8643 | train 5.2s
Epoch 405 loss  95.8338 | train 5.3s | Acc 19.09% | infer 18.7s
Epoch 406 loss  94.7681 | train 5.2s
Epoch 407 loss  93.8631 | train 5.2s
Epoch 408 loss  92.7204 | train 5.2s
Epoch 409 loss  91.7375 | train 5.2s
Epoch 410 loss  90.6926 | train 5.2s | Acc 18.98% | infer 18.7s
Epoch 411 loss  89.7260 | train 5.2s
Epoch 412 loss  88.7205 | train 5.2s
Epoch 413 loss  87.8790 | train 5.2s
Epoch 414 loss  86.8857 | train 5.2s
Epoch 415 loss  85.9109 | train 5.2s | Acc 19.61% | infer 18.7s
Epoch 416 loss  84.8593 | train 5.2s
Epoch 417 loss  83.9103 | train 5.2s
Epoch 418 loss  95.9680 | train 5.2s
Epoch 419 loss  90.4503 | train 5.2s
Epoch 420 loss  85.9813 | train 5.2s | Acc 3.74% | infer 18.7s
Epoch 421 loss  83.1608 | train 5.2s
Epoch 422 loss  81.8033 | train 5.2s
Epoch 423 loss  80.3637 | train 5.3s
Epoch 424 loss  79.2520 | train 5.2s
Epoch 425 loss  78.2397 | train 5.2s | Acc 9.16% | infer 18.7s
Epoch 426 loss  77.3426 | train 5.2s
Epoch 427 loss  76.4403 | train 5.2s
Epoch 428 loss  75.5567 | train 5.2s
Epoch 429 loss  74.6507 | train 5.2s
Epoch 430 loss  73.8326 | train 5.2s | Acc 11.27% | infer 18.7s
Epoch 431 loss  73.0415 | train 5.2s
Epoch 432 loss  72.1702 | train 5.2s
Epoch 433 loss  71.2539 | train 5.2s
Epoch 434 loss  70.4988 | train 5.2s
Epoch 435 loss  69.7080 | train 5.2s | Acc 15.09% | infer 18.7s
Epoch 436 loss  69.0156 | train 5.3s
Epoch 437 loss  68.2855 | train 5.2s
Epoch 438 loss  67.5185 | train 5.2s
Epoch 439 loss  66.8476 | train 5.2s
Epoch 440 loss  65.9651 | train 5.2s | Acc 16.15% | infer 18.7s
Epoch 441 loss  65.2857 | train 5.2s
Epoch 442 loss  64.5981 | train 5.2s
Epoch 443 loss  63.8494 | train 5.2s
Epoch 444 loss  63.1547 | train 5.2s
Epoch 445 loss  62.5299 | train 5.2s | Acc 15.42% | infer 18.7s
Epoch 446 loss  61.6796 | train 5.2s
Epoch 447 loss  61.0194 | train 5.2s
Epoch 448 loss  60.4430 | train 5.2s
Epoch 449 loss  59.6997 | train 5.2s
Epoch 450 loss  59.0162 | train 5.2s | Acc 16.84% | infer 18.7s
Epoch 451 loss  58.4363 | train 5.2s
Epoch 452 loss  57.5778 | train 5.2s
Epoch 453 loss  57.1562 | train 5.2s
Epoch 454 loss  56.5216 | train 5.2s
Epoch 455 loss  55.8608 | train 5.2s | Acc 16.90% | infer 18.7s
Epoch 456 loss  55.2176 | train 5.2s
Epoch 457 loss  54.6073 | train 5.2s
Epoch 458 loss  53.9583 | train 5.2s
Epoch 459 loss  53.3797 | train 5.2s
Epoch 460 loss  52.8224 | train 5.2s | Acc 17.00% | infer 18.7s
Epoch 461 loss  52.1201 | train 5.2s
Epoch 462 loss  51.5615 | train 5.2s
Epoch 463 loss  50.9621 | train 5.2s
Epoch 464 loss  50.4150 | train 5.3s
Epoch 465 loss  49.9146 | train 5.2s | Acc 19.10% | infer 18.6s
Epoch 466 loss  49.2854 | train 5.2s
Epoch 467 loss  48.7561 | train 5.2s
Epoch 468 loss  48.2016 | train 5.3s
Epoch 469 loss  47.6866 | train 5.2s
Epoch 470 loss  47.0939 | train 5.1s | Acc 19.04% | infer 18.7s
Epoch 471 loss  46.5889 | train 5.2s
Epoch 472 loss  46.0448 | train 5.2s
Epoch 473 loss  45.5140 | train 5.2s
Epoch 474 loss  45.0166 | train 5.2s
Epoch 475 loss  44.4533 | train 5.2s | Acc 18.87% | infer 18.7s
Epoch 476 loss  43.9936 | train 5.2s
Epoch 477 loss  43.4744 | train 5.2s
Epoch 478 loss  42.9527 | train 5.2s
Epoch 479 loss  42.5549 | train 5.2s
Epoch 480 loss  41.9996 | train 5.2s | Acc 18.57% | infer 18.7s
Epoch 481 loss  41.4883 | train 5.2s
Epoch 482 loss  40.9432 | train 5.2s
Epoch 483 loss  40.6047 | train 5.2s
Epoch 484 loss  40.0963 | train 5.2s
Epoch 485 loss  39.6060 | train 5.2s | Acc 19.23% | infer 18.7s
Epoch 486 loss  39.2044 | train 5.2s
Epoch 487 loss  38.7125 | train 5.2s
Epoch 488 loss  38.2647 | train 5.2s
Epoch 489 loss  37.8587 | train 5.2s
Epoch 490 loss  37.3275 | train 5.3s | Acc 20.38% | infer 18.7s
Epoch 491 loss  36.9726 | train 5.2s
Epoch 492 loss  36.5658 | train 5.2s
Epoch 493 loss  36.0927 | train 5.2s
Epoch 494 loss  35.6708 | train 5.2s
Epoch 495 loss  35.2429 | train 5.2s | Acc 19.02% | infer 18.7s
Epoch 496 loss  34.7677 | train 5.2s
Epoch 497 loss  34.3527 | train 5.2s
Epoch 498 loss  34.0314 | train 5.2s
Epoch 499 loss  33.5496 | train 5.2s
Epoch 500 loss  33.1471 | train 5.2s | Acc 21.01% | infer 18.6s
Epoch 501 loss  32.7547 | train 5.2s
Epoch 502 loss  32.3819 | train 5.2s
Epoch 503 loss  32.0858 | train 5.2s
Epoch 504 loss  31.6671 | train 5.2s
Epoch 505 loss  31.2667 | train 5.2s | Acc 20.99% | infer 18.7s
Epoch 506 loss  30.9148 | train 5.2s
Epoch 507 loss  30.4704 | train 5.2s
Epoch 508 loss  30.1413 | train 5.2s
Epoch 509 loss  29.7963 | train 5.2s
Epoch 510 loss  29.3746 | train 5.2s | Acc 20.72% | infer 18.7s
Epoch 511 loss  29.0555 | train 5.2s
Epoch 512 loss  28.7329 | train 5.2s
Epoch 513 loss  28.3933 | train 5.2s
Epoch 514 loss  28.1080 | train 5.3s
Epoch 515 loss  27.7098 | train 5.2s | Acc 21.33% | infer 18.7s
Epoch 516 loss  27.3745 | train 5.3s
Epoch 517 loss  27.1051 | train 5.3s
Epoch 518 loss  26.6946 | train 5.2s
Epoch 519 loss  26.4233 | train 5.2s
Epoch 520 loss  26.0593 | train 5.2s | Acc 20.64% | infer 18.6s
Epoch 521 loss  25.7586 | train 5.3s
Epoch 522 loss  25.4669 | train 5.2s
Epoch 523 loss  25.1640 | train 5.2s
Epoch 524 loss  24.8530 | train 5.2s
Epoch 525 loss  24.5506 | train 5.2s | Acc 21.47% | infer 18.7s
Epoch 526 loss  24.2463 | train 5.2s
Epoch 527 loss  23.9163 | train 5.2s
Epoch 528 loss  23.6805 | train 5.2s
Epoch 529 loss  23.3534 | train 5.2s
Epoch 530 loss  23.0992 | train 5.2s | Acc 20.93% | infer 18.7s
Epoch 531 loss  22.8225 | train 5.2s
Epoch 532 loss  22.5148 | train 5.2s
Epoch 533 loss  22.2128 | train 5.2s
Epoch 534 loss  21.9587 | train 5.2s
Epoch 535 loss  21.7053 | train 5.2s | Acc 22.05% | infer 18.7s
Epoch 536 loss  21.4570 | train 5.2s
Epoch 537 loss  21.1442 | train 5.2s
Epoch 538 loss  20.9145 | train 5.3s
Epoch 539 loss  20.6088 | train 5.3s
Epoch 540 loss  20.3885 | train 5.2s | Acc 22.96% | infer 18.7s
Epoch 541 loss  20.1648 | train 5.3s
Epoch 542 loss  19.8603 | train 5.2s
Epoch 543 loss  19.6416 | train 5.2s
Epoch 544 loss  19.3925 | train 5.2s
Epoch 545 loss  19.1694 | train 5.2s | Acc 22.86% | infer 18.7s
Epoch 546 loss  18.8710 | train 5.2s
Epoch 547 loss  18.6559 | train 5.2s
Epoch 548 loss  18.4105 | train 5.2s
Epoch 549 loss  18.1272 | train 5.2s
Epoch 550 loss  17.9971 | train 5.2s | Acc 24.11% | infer 18.7s
Epoch 551 loss  17.6973 | train 5.2s
Epoch 552 loss  17.4860 | train 5.2s
Epoch 553 loss  17.3114 | train 5.2s
Epoch 554 loss  17.0846 | train 5.2s
Epoch 555 loss  16.8390 | train 5.2s | Acc 24.03% | infer 18.7s
Epoch 556 loss  16.6393 | train 5.2s
Epoch 557 loss  16.4337 | train 5.2s
Epoch 558 loss  16.2166 | train 5.2s
Epoch 559 loss  16.0486 | train 5.2s
Epoch 560 loss  15.8136 | train 5.2s | Acc 24.52% | infer 18.6s
Epoch 561 loss  15.5981 | train 5.2s
Epoch 562 loss  15.4186 | train 5.2s
Epoch 563 loss  15.2340 | train 5.2s
Epoch 564 loss  15.0040 | train 5.3s
Epoch 565 loss  15.2356 | train 5.2s | Acc 19.71% | infer 18.7s
Epoch 566 loss  15.3575 | train 5.2s
Epoch 567 loss  14.9410 | train 5.2s
Epoch 568 loss  14.4532 | train 5.2s
Epoch 569 loss  14.1981 | train 5.2s
Epoch 570 loss  14.0036 | train 5.2s | Acc 22.99% | infer 18.7s
Epoch 571 loss  13.7726 | train 5.1s
Epoch 572 loss  13.6042 | train 5.2s
Epoch 573 loss  13.4182 | train 5.2s
Epoch 574 loss  13.1963 | train 5.2s
Epoch 575 loss  13.0697 | train 5.2s | Acc 25.95% | infer 18.6s
Epoch 576 loss  12.8431 | train 5.2s
Epoch 577 loss  12.7366 | train 5.2s
Epoch 578 loss  12.5325 | train 5.2s
Epoch 579 loss  12.4004 | train 5.3s
Epoch 580 loss  12.2353 | train 5.2s | Acc 25.91% | infer 18.7s
Epoch 581 loss  12.0779 | train 5.2s
Epoch 582 loss  11.9268 | train 5.2s
Epoch 583 loss  11.7913 | train 5.2s
Epoch 584 loss  11.5786 | train 5.2s
Epoch 585 loss  11.4301 | train 5.2s | Acc 26.40% | infer 18.7s
Epoch 586 loss  11.3031 | train 5.3s
Epoch 587 loss  11.1425 | train 5.3s
Epoch 588 loss  10.9963 | train 5.3s
Epoch 589 loss  10.8706 | train 5.2s
Epoch 590 loss  10.7387 | train 5.2s | Acc 27.31% | infer 18.7s
Epoch 591 loss  10.6067 | train 5.2s
Epoch 592 loss  10.4511 | train 5.3s
Epoch 593 loss  10.3090 | train 5.2s
Epoch 594 loss  10.1635 | train 5.2s
Epoch 595 loss  10.0352 | train 5.1s | Acc 25.78% | infer 18.7s
Epoch 596 loss   9.9206 | train 5.2s
Epoch 597 loss   9.7785 | train 5.2s
Epoch 598 loss   9.6477 | train 5.3s
Epoch 599 loss   9.5542 | train 5.2s
Epoch 600 loss   9.4096 | train 5.2s | Acc 27.28% | infer 18.7s
Epoch 601 loss   9.2704 | train 5.2s
Epoch 602 loss   9.1563 | train 5.2s
Epoch 603 loss   9.0592 | train 5.2s
Epoch 604 loss   8.9266 | train 5.2s
Epoch 605 loss   8.7994 | train 5.2s | Acc 28.03% | infer 18.7s
Epoch 606 loss   8.7219 | train 5.2s
Epoch 607 loss   8.5811 | train 5.2s
Epoch 608 loss   8.4796 | train 5.2s
Epoch 609 loss   8.3361 | train 5.2s
Epoch 610 loss   8.1814 | train 5.2s | Acc 27.93% | infer 18.7s
Epoch 611 loss   8.1048 | train 5.2s
Epoch 612 loss   8.0056 | train 5.2s
Epoch 613 loss   7.9225 | train 5.2s
Epoch 614 loss   7.8296 | train 5.2s
Epoch 615 loss   7.7082 | train 5.2s | Acc 28.67% | infer 18.7s
Epoch 616 loss   7.5787 | train 5.2s
Epoch 617 loss   7.4711 | train 5.2s
Epoch 618 loss   7.4159 | train 5.2s
Epoch 619 loss   7.2958 | train 5.2s
Epoch 620 loss   7.1576 | train 5.2s | Acc 27.73% | infer 18.6s
Epoch 621 loss   7.1130 | train 5.2s
Epoch 622 loss   6.9960 | train 5.2s
Epoch 623 loss   6.9121 | train 5.2s
Epoch 624 loss   6.8122 | train 5.2s
Epoch 625 loss   6.7174 | train 5.2s | Acc 29.09% | infer 18.6s
Epoch 626 loss   6.6445 | train 5.2s
Epoch 627 loss   6.5455 | train 5.2s
Epoch 628 loss   6.4501 | train 5.2s
Epoch 629 loss   6.3379 | train 5.2s
Epoch 630 loss   6.2772 | train 5.2s | Acc 28.51% | infer 18.7s
Epoch 631 loss   6.1912 | train 5.2s
Epoch 632 loss   6.1194 | train 5.2s
Epoch 633 loss   6.0280 | train 5.2s
Epoch 634 loss   5.9394 | train 5.2s
Epoch 635 loss   6.0617 | train 5.2s | Acc 27.65% | infer 18.7s
Epoch 636 loss   5.9470 | train 5.2s
Epoch 637 loss   5.7981 | train 5.2s
Epoch 638 loss   5.6625 | train 5.2s
Epoch 639 loss   5.5872 | train 5.2s
Epoch 640 loss   5.5153 | train 5.2s | Acc 29.55% | infer 18.7s
Epoch 641 loss   5.4131 | train 5.2s
Epoch 642 loss   5.4141 | train 5.2s
Epoch 643 loss   5.2929 | train 5.2s
Epoch 644 loss   5.2090 | train 5.3s
Epoch 645 loss   5.1201 | train 5.2s | Acc 29.65% | infer 18.6s
Epoch 646 loss   5.0459 | train 5.2s
Epoch 647 loss   4.9667 | train 5.2s
Epoch 648 loss   4.9207 | train 5.2s
Epoch 649 loss   4.8616 | train 5.2s
Epoch 650 loss   4.7766 | train 5.2s | Acc 30.61% | infer 18.6s
Epoch 651 loss   4.7116 | train 5.2s
Epoch 652 loss   4.6187 | train 5.2s
Epoch 653 loss   4.5686 | train 5.2s
Epoch 654 loss   4.5135 | train 5.2s
Epoch 655 loss   4.4466 | train 5.2s | Acc 31.19% | infer 18.6s
Epoch 656 loss   4.3975 | train 5.2s
Epoch 657 loss   4.3156 | train 5.2s
Epoch 658 loss   4.2664 | train 5.2s
Epoch 659 loss   4.2221 | train 5.2s
Epoch 660 loss   4.1489 | train 5.2s | Acc 31.02% | infer 18.6s
Epoch 661 loss   4.0763 | train 5.2s
Epoch 662 loss   4.0473 | train 5.2s
Epoch 663 loss   3.9756 | train 5.2s
Epoch 664 loss   3.9219 | train 5.2s
Epoch 665 loss   3.8701 | train 5.2s | Acc 30.49% | infer 18.7s
Epoch 666 loss   3.8036 | train 5.3s
Epoch 667 loss   3.7720 | train 5.2s
Epoch 668 loss   3.6910 | train 5.2s
Epoch 669 loss   3.6609 | train 5.2s
Epoch 670 loss   3.5989 | train 5.2s | Acc 31.43% | infer 18.7s
Epoch 671 loss   3.5475 | train 5.2s
Epoch 672 loss   3.4955 | train 5.2s
Epoch 673 loss   3.4377 | train 5.1s
Epoch 674 loss   3.4107 | train 5.2s
Epoch 675 loss   3.3547 | train 5.2s | Acc 32.39% | infer 18.6s
Epoch 676 loss   3.3094 | train 5.2s
Epoch 677 loss   3.2652 | train 5.2s
Epoch 678 loss   3.2128 | train 5.2s
Epoch 679 loss   3.1595 | train 5.2s
Epoch 680 loss   3.1088 | train 5.2s | Acc 33.08% | infer 18.7s
Epoch 681 loss   3.0839 | train 5.2s
Epoch 682 loss   3.0303 | train 5.2s
Epoch 683 loss   2.9873 | train 5.2s
Epoch 684 loss   2.9373 | train 5.2s
Epoch 685 loss   2.9141 | train 5.1s | Acc 32.77% | infer 18.7s
Epoch 686 loss   2.8580 | train 5.2s
Epoch 687 loss   2.8112 | train 5.2s
Epoch 688 loss   2.7694 | train 5.2s
Epoch 689 loss   2.7420 | train 5.1s
Epoch 690 loss   2.6981 | train 5.2s | Acc 33.72% | infer 18.7s
Epoch 691 loss   2.6517 | train 5.2s
Epoch 692 loss   2.6134 | train 5.2s
Epoch 693 loss   2.5795 | train 5.2s
Epoch 694 loss   2.5414 | train 5.2s
Epoch 695 loss   2.4953 | train 5.2s | Acc 33.60% | infer 18.7s
Epoch 696 loss   2.4696 | train 5.2s
Epoch 697 loss   2.4417 | train 5.2s
Epoch 698 loss   2.4024 | train 5.2s
Epoch 699 loss   2.3696 | train 5.2s
Epoch 700 loss   2.3416 | train 5.2s | Acc 33.94% | infer 18.7s
Epoch 701 loss   2.3387 | train 5.2s
Epoch 702 loss   2.2732 | train 5.2s
Epoch 703 loss   2.2222 | train 5.2s
Epoch 704 loss   2.1986 | train 5.2s
Epoch 705 loss   2.1629 | train 5.2s | Acc 34.01% | infer 18.7s
Epoch 706 loss   2.1176 | train 5.3s
Epoch 707 loss   2.0961 | train 5.2s
Epoch 708 loss   2.1246 | train 5.2s
Epoch 709 loss   2.0776 | train 5.2s
Epoch 710 loss   2.0168 | train 5.2s | Acc 34.49% | infer 18.7s
Epoch 711 loss   1.9812 | train 5.2s
Epoch 712 loss   1.9602 | train 5.2s
Epoch 713 loss   1.9327 | train 5.2s
Epoch 714 loss   1.8872 | train 5.2s
Epoch 715 loss   1.8705 | train 5.2s | Acc 34.14% | infer 18.7s
Epoch 716 loss   1.8457 | train 5.2s
Epoch 717 loss   1.8120 | train 5.3s
Epoch 718 loss   1.7943 | train 5.2s
Epoch 719 loss   1.7573 | train 5.2s
Epoch 720 loss   1.7334 | train 5.2s | Acc 34.85% | infer 18.7s
Epoch 721 loss   1.7060 | train 5.2s
Epoch 722 loss   1.6982 | train 5.2s
Epoch 723 loss   1.6595 | train 5.3s
Epoch 724 loss   1.6274 | train 5.2s
Epoch 725 loss   1.6080 | train 5.2s | Acc 35.31% | infer 18.7s
Epoch 726 loss   1.5832 | train 5.2s
Epoch 727 loss   1.5660 | train 5.2s
Epoch 728 loss   1.5453 | train 5.2s
Epoch 729 loss   1.5139 | train 5.3s
Epoch 730 loss   1.5005 | train 5.2s | Acc 35.27% | infer 18.7s
Epoch 731 loss   1.4795 | train 5.3s
Epoch 732 loss   1.4484 | train 5.3s
Epoch 733 loss   1.4368 | train 5.2s
Epoch 734 loss   1.4154 | train 5.2s
Epoch 735 loss   1.3817 | train 5.2s | Acc 35.95% | infer 18.7s
Epoch 736 loss   1.3624 | train 5.2s
Epoch 737 loss   1.3468 | train 5.2s
Epoch 738 loss   1.3190 | train 5.1s
Epoch 739 loss   1.2946 | train 5.2s
Epoch 740 loss   1.2776 | train 5.2s | Acc 34.79% | infer 18.7s
Epoch 741 loss   1.2648 | train 5.2s
Epoch 742 loss   1.2455 | train 5.2s
Epoch 743 loss   1.2345 | train 5.2s
Epoch 744 loss   1.2135 | train 5.2s
Epoch 745 loss   1.1861 | train 5.2s | Acc 35.75% | infer 18.7s
Epoch 746 loss   1.1748 | train 5.2s
Epoch 747 loss   1.1491 | train 5.3s
Epoch 748 loss   1.1409 | train 5.2s
Epoch 749 loss   1.1241 | train 5.3s
Epoch 750 loss   1.1113 | train 5.2s | Acc 36.02% | infer 18.7s
Epoch 751 loss   1.0936 | train 5.2s
Epoch 752 loss   1.0934 | train 5.2s
Epoch 753 loss   1.0638 | train 5.2s
Epoch 754 loss   1.0399 | train 5.2s
Epoch 755 loss   1.0268 | train 5.2s | Acc 35.80% | infer 18.7s
Epoch 756 loss   1.0115 | train 5.2s
Epoch 757 loss   0.9920 | train 5.3s
Epoch 758 loss   0.9777 | train 5.2s
Epoch 759 loss   0.9663 | train 5.2s
Epoch 760 loss   0.9484 | train 5.3s | Acc 36.12% | infer 18.6s
Epoch 761 loss   0.9403 | train 5.2s
Epoch 762 loss   0.9251 | train 5.2s
Epoch 763 loss   0.9014 | train 5.2s
Epoch 764 loss   0.8896 | train 5.2s
Epoch 765 loss   0.8739 | train 5.2s | Acc 36.93% | infer 18.7s
Epoch 766 loss   0.8677 | train 5.2s
Epoch 767 loss   0.8592 | train 5.2s
Epoch 768 loss   0.8512 | train 5.3s
Epoch 769 loss   0.8334 | train 5.2s
Epoch 770 loss   0.8178 | train 5.2s | Acc 36.89% | infer 18.7s
Epoch 771 loss   0.8082 | train 5.2s
Epoch 772 loss   0.7923 | train 5.2s
Epoch 773 loss   0.7776 | train 5.2s
Epoch 774 loss   0.7703 | train 5.2s
Epoch 775 loss   0.7553 | train 5.3s | Acc 36.41% | infer 18.7s
Epoch 776 loss   0.7418 | train 5.2s
Epoch 777 loss   0.7332 | train 5.2s
Epoch 778 loss   0.7324 | train 5.2s
Epoch 779 loss   0.7230 | train 5.3s
Epoch 780 loss   0.7097 | train 5.3s | Acc 36.67% | infer 18.6s
Epoch 781 loss   0.7005 | train 5.2s
Epoch 782 loss   0.6799 | train 5.2s
Epoch 783 loss   0.6613 | train 5.3s
Epoch 784 loss   0.6570 | train 5.2s
Epoch 785 loss   0.6541 | train 5.2s | Acc 37.51% | infer 18.7s
Epoch 786 loss   0.6406 | train 5.2s
Epoch 787 loss   0.6294 | train 5.2s
Epoch 788 loss   0.6202 | train 5.3s
Epoch 789 loss   0.6154 | train 5.2s
Epoch 790 loss   0.6104 | train 5.2s | Acc 37.14% | infer 18.7s
Epoch 791 loss   0.6010 | train 5.3s
Epoch 792 loss   0.5935 | train 5.4s
Epoch 793 loss   0.5833 | train 5.3s
Epoch 794 loss   0.5810 | train 5.3s
Epoch 795 loss   0.5684 | train 5.2s | Acc 37.80% | infer 18.8s
Epoch 796 loss   0.5575 | train 5.2s
Epoch 797 loss   0.5527 | train 5.2s
Epoch 798 loss   0.5459 | train 5.3s
Epoch 799 loss   0.5407 | train 5.2s
Epoch 800 loss   0.5343 | train 5.2s | Acc 38.38% | infer 18.7s
Epoch 801 loss   0.5133 | train 5.3s
Epoch 802 loss   0.5083 | train 5.2s
Epoch 803 loss   0.5004 | train 5.3s
Epoch 804 loss   0.4976 | train 5.3s
Epoch 805 loss   0.4879 | train 5.3s | Acc 38.20% | infer 18.7s
Epoch 806 loss   0.4769 | train 5.3s
Epoch 807 loss   0.4679 | train 5.3s
Epoch 808 loss   0.4663 | train 5.3s
Epoch 809 loss   0.4543 | train 5.2s
Epoch 810 loss   0.4425 | train 5.3s | Acc 38.56% | infer 18.7s
Epoch 811 loss   0.4406 | train 5.2s
Epoch 812 loss   0.4283 | train 5.2s
Epoch 813 loss   0.4238 | train 5.2s
Epoch 814 loss   0.4191 | train 5.3s
Epoch 815 loss   0.4175 | train 5.3s | Acc 37.80% | infer 18.7s
Epoch 816 loss   0.4119 | train 5.3s
Epoch 817 loss   0.4196 | train 5.2s
Epoch 818 loss   0.4093 | train 5.3s
Epoch 819 loss   0.3973 | train 5.2s
Epoch 820 loss   0.3875 | train 5.3s | Acc 37.61% | infer 18.7s
Epoch 821 loss   0.3781 | train 5.4s
Epoch 822 loss   0.3760 | train 5.3s
Epoch 823 loss   0.3691 | train 5.3s
Epoch 824 loss   0.3597 | train 5.2s
Epoch 825 loss   0.3583 | train 5.4s | Acc 38.83% | infer 18.7s
Epoch 826 loss   0.3511 | train 5.3s
Epoch 827 loss   0.3477 | train 5.2s
Epoch 828 loss   0.3477 | train 5.2s
Epoch 829 loss   0.3360 | train 5.2s
Epoch 830 loss   0.3301 | train 5.3s | Acc 37.94% | infer 18.7s
Epoch 831 loss   0.3248 | train 5.3s
Epoch 832 loss   0.3247 | train 5.2s
Epoch 833 loss   0.3160 | train 5.2s
Epoch 834 loss   0.3122 | train 5.2s
Epoch 835 loss   0.3036 | train 5.2s | Acc 36.81% | infer 18.7s
Epoch 836 loss   0.3003 | train 5.2s
Epoch 837 loss   0.3130 | train 5.3s
Epoch 838 loss   0.3048 | train 5.3s
Epoch 839 loss   0.2966 | train 5.2s
Epoch 840 loss   0.2894 | train 5.2s | Acc 36.43% | infer 18.7s
Epoch 841 loss   0.2835 | train 5.3s
Epoch 842 loss   0.2837 | train 5.2s
Epoch 843 loss   0.2778 | train 5.3s
Epoch 844 loss   0.2775 | train 5.2s
Epoch 845 loss   0.2717 | train 5.1s | Acc 37.91% | infer 18.7s
Epoch 846 loss   0.2687 | train 5.2s
Epoch 847 loss   0.2566 | train 5.2s
Epoch 848 loss   0.2561 | train 5.2s
Epoch 849 loss   0.2536 | train 5.1s
Epoch 850 loss   0.2481 | train 5.2s | Acc 38.23% | infer 18.7s
Epoch 851 loss   0.2455 | train 5.2s
Epoch 852 loss   0.2424 | train 5.2s
Epoch 853 loss   0.2414 | train 5.2s
Epoch 854 loss   0.2421 | train 5.2s
Epoch 855 loss   0.2342 | train 5.3s | Acc 38.62% | infer 18.6s
Epoch 856 loss   0.2295 | train 5.2s
Epoch 857 loss   0.2267 | train 5.2s
Epoch 858 loss   0.2183 | train 5.2s
Epoch 859 loss   0.2178 | train 5.2s
Epoch 860 loss   0.2120 | train 5.2s | Acc 37.85% | infer 18.6s
Epoch 861 loss   0.2145 | train 5.2s
Epoch 862 loss   0.2103 | train 5.2s
Epoch 863 loss   0.2085 | train 5.1s
Epoch 864 loss   0.2024 | train 5.2s
Epoch 865 loss   0.2001 | train 5.2s | Acc 37.02% | infer 18.7s
Epoch 866 loss   0.1966 | train 5.3s
Epoch 867 loss   0.1899 | train 5.3s
Epoch 868 loss   0.1901 | train 5.2s
Epoch 869 loss   0.1887 | train 5.3s
Epoch 870 loss   0.1851 | train 5.2s | Acc 37.83% | infer 18.6s
Epoch 871 loss   0.1821 | train 5.2s
Epoch 872 loss   0.1822 | train 5.2s
Epoch 873 loss   0.1840 | train 5.2s
Epoch 874 loss   0.1812 | train 5.2s
Epoch 875 loss   0.1793 | train 5.2s | Acc 38.03% | infer 18.6s
Epoch 876 loss   0.1803 | train 5.3s
Epoch 877 loss   0.1814 | train 5.2s
Epoch 878 loss   0.1765 | train 5.2s
Epoch 879 loss   0.1726 | train 5.2s
Epoch 880 loss   0.1671 | train 5.3s | Acc 37.02% | infer 18.7s
Epoch 881 loss   0.1631 | train 5.2s
Epoch 882 loss   0.1610 | train 5.3s
Epoch 883 loss   0.1611 | train 5.2s
Epoch 884 loss   0.1585 | train 5.2s
Epoch 885 loss   0.1579 | train 5.2s | Acc 37.89% | infer 18.6s
Epoch 886 loss   0.1582 | train 5.2s
Epoch 887 loss   0.1543 | train 5.3s
Epoch 888 loss   0.1522 | train 5.2s
Epoch 889 loss   0.1460 | train 5.2s
Epoch 890 loss   0.1465 | train 5.2s | Acc 38.49% | infer 18.6s
Epoch 891 loss   0.1495 | train 5.2s
Epoch 892 loss   0.1430 | train 5.2s
Epoch 893 loss   0.1448 | train 5.2s
Epoch 894 loss   0.1381 | train 5.2s
Epoch 895 loss   0.1372 | train 5.2s | Acc 36.72% | infer 18.6s
Epoch 896 loss   0.1363 | train 5.2s
Epoch 897 loss   0.1362 | train 5.2s
Epoch 898 loss   0.1336 | train 5.2s
Epoch 899 loss   0.1470 | train 5.2s
Epoch 900 loss   0.1347 | train 5.2s | Acc 36.97% | infer 18.7s
Epoch 901 loss   0.1277 | train 5.2s
Epoch 902 loss   0.1303 | train 5.3s
Epoch 903 loss   0.1267 | train 5.2s
Epoch 904 loss   0.1251 | train 5.2s
Epoch 905 loss   0.1231 | train 5.3s | Acc 37.21% | infer 18.7s
Epoch 906 loss   0.1202 | train 5.2s
Epoch 907 loss   0.1197 | train 5.2s
Epoch 908 loss   0.1167 | train 5.2s
Epoch 909 loss   0.1158 | train 5.3s
Epoch 910 loss   0.1141 | train 5.2s | Acc 37.87% | infer 18.7s
Epoch 911 loss   0.1130 | train 5.2s
Epoch 912 loss   0.1157 | train 5.2s
Epoch 913 loss   0.1145 | train 5.3s
Epoch 914 loss   0.1124 | train 5.2s
Epoch 915 loss   0.1077 | train 5.2s | Acc 35.94% | infer 18.7s
Epoch 916 loss   0.1071 | train 5.3s
Epoch 917 loss   0.1059 | train 5.2s
Epoch 918 loss   0.1010 | train 5.2s
Epoch 919 loss   0.1024 | train 5.2s
Epoch 920 loss   0.1030 | train 5.2s | Acc 38.04% | infer 18.7s
Epoch 921 loss   0.1063 | train 5.2s
Epoch 922 loss   0.1101 | train 5.2s
Epoch 923 loss   0.1014 | train 5.2s
Epoch 924 loss   0.1024 | train 5.3s
Epoch 925 loss   0.1027 | train 5.2s | Acc 34.98% | infer 18.7s
Epoch 926 loss   0.1011 | train 5.2s
Epoch 927 loss   0.0993 | train 5.2s
Epoch 928 loss   0.1014 | train 5.2s
Epoch 929 loss   0.1002 | train 5.2s
Epoch 930 loss   0.1013 | train 5.2s | Acc 35.80% | infer 18.7s
Epoch 931 loss   0.1005 | train 5.2s
Epoch 932 loss   0.0950 | train 5.2s
Epoch 933 loss   0.0906 | train 5.2s
Epoch 934 loss   0.0887 | train 5.2s
Epoch 935 loss   0.0911 | train 5.2s | Acc 34.39% | infer 18.7s
Epoch 936 loss   0.0878 | train 5.2s
Epoch 937 loss   0.0870 | train 5.2s
Epoch 938 loss   0.0826 | train 5.2s
Epoch 939 loss   0.0816 | train 5.2s
Epoch 940 loss   0.0829 | train 5.2s | Acc 35.39% | infer 18.7s
Epoch 941 loss   0.0838 | train 5.2s
Epoch 942 loss   0.0819 | train 5.2s
Epoch 943 loss   0.0792 | train 5.2s
Epoch 944 loss   0.0744 | train 5.2s
Epoch 945 loss   0.0752 | train 5.2s | Acc 35.50% | infer 18.7s
Epoch 946 loss   0.0757 | train 5.2s
Epoch 947 loss   0.0741 | train 5.3s
Epoch 948 loss   0.0909 | train 5.2s
Epoch 949 loss   0.0989 | train 5.3s
Epoch 950 loss   0.0864 | train 5.2s | Acc 33.36% | infer 18.6s
Epoch 951 loss   0.0775 | train 5.2s
Epoch 952 loss   0.0762 | train 5.2s
Epoch 953 loss   0.0747 | train 5.2s
Epoch 954 loss   0.0700 | train 5.2s
Epoch 955 loss   0.0686 | train 5.2s | Acc 35.73% | infer 18.6s
Epoch 956 loss   0.0684 | train 5.2s
Epoch 957 loss   0.0670 | train 5.2s
Epoch 958 loss   0.0691 | train 5.2s
Epoch 959 loss   0.0685 | train 5.2s
Epoch 960 loss   0.0668 | train 5.2s | Acc 36.78% | infer 18.7s
Epoch 961 loss   0.0657 | train 5.2s
Epoch 962 loss   0.0653 | train 5.3s
Epoch 963 loss   0.0655 | train 5.2s
Epoch 964 loss   0.0667 | train 5.2s
Epoch 965 loss   0.0663 | train 5.2s | Acc 34.74% | infer 18.7s
Epoch 966 loss   0.0634 | train 5.2s
Epoch 967 loss   0.0605 | train 5.2s
Epoch 968 loss   0.0563 | train 5.2s
Epoch 969 loss   0.0584 | train 5.2s
Epoch 970 loss   0.0571 | train 5.3s | Acc 33.93% | infer 18.6s
Epoch 971 loss   0.0549 | train 5.2s
Epoch 972 loss   0.0541 | train 5.2s
Epoch 973 loss   0.0543 | train 5.2s
Epoch 974 loss   0.0556 | train 5.3s
Epoch 975 loss   0.0568 | train 5.3s | Acc 31.04% | infer 18.7s
Epoch 976 loss   0.0621 | train 5.2s
Epoch 977 loss   0.0603 | train 5.2s
Epoch 978 loss   0.0622 | train 5.2s
Epoch 979 loss   0.0584 | train 5.2s
Epoch 980 loss   0.0561 | train 5.2s | Acc 32.52% | infer 18.7s
Epoch 981 loss   0.0576 | train 5.3s
Epoch 982 loss   0.0546 | train 5.2s
Epoch 983 loss   0.0540 | train 5.2s
Epoch 984 loss   0.0527 | train 5.2s
Epoch 985 loss   0.0557 | train 5.3s | Acc 32.12% | infer 18.7s
Epoch 986 loss   0.0536 | train 5.3s
Epoch 987 loss   0.0563 | train 5.2s
Epoch 988 loss   0.0587 | train 5.3s
Epoch 989 loss   0.0541 | train 5.3s
Epoch 990 loss   0.0517 | train 5.2s | Acc 37.21% | infer 18.7s
Epoch 991 loss   0.0492 | train 5.3s
Epoch 992 loss   0.0478 | train 5.2s
Epoch 993 loss   0.0515 | train 5.2s
Epoch 994 loss   0.0499 | train 5.2s
Epoch 995 loss   0.0512 | train 5.3s | Acc 35.82% | infer 18.7s
Epoch 996 loss   0.0486 | train 5.2s
Epoch 997 loss   0.0496 | train 5.2s
Epoch 998 loss   0.0490 | train 5.2s
Epoch 999 loss   0.0485 | train 5.2s
Epoch 1000 loss   0.0487 | train 5.2s | Acc 38.05% | infer 18.7s
Final Heun multi-T eval:
Heun T=  2 acc 1.00% | infer 1.5s
Heun T=  5 acc 1.00% | infer 2.8s
Heun T= 10 acc 1.00% | infer 5.1s
Heun T= 20 acc 37.38% | infer 9.6s
Heun T= 40 acc 37.94% | infer 18.6s
Heun T= 80 acc 38.67% | infer 36.7s
```
