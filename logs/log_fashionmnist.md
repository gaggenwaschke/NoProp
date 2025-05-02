
## FashionMNIST ( 10 categories) with ResNet-18 module + fused cls header
- with flow matching 
- avg_loss : 1 epoch average loss , l : total loss, f : flow loss, a : anchor loss(label), kl : kl loss

```
start Namespace(dataset='fashionmnist', data_root='./data', backbone='resnet18', time_dim=64, embed_dim=800, epochs=1000, batch_size=4096, init_type='random', lambda_anchor=5.0)

Epoch    1/1000 done | avg_loss 393.7097  l:389.0295, f:  0.2141, a:  0.0311,  kl:388.6599  in 3.7s  
Epoch    2/1000 done | avg_loss 383.9871  l:380.2843, f:  0.1980, a:  0.0197,  kl:379.9877  in 2.9s  
Epoch    3/1000 done | avg_loss 375.3801  l:371.7953, f:  0.1905, a:  0.0164,  kl:371.5227  in 2.9s  
Epoch    4/1000 done | avg_loss 367.0218  l:363.3329, f:  0.1815, a:  0.0146,  kl:363.0784  in 2.9s  
Epoch    5/1000 done | avg_loss 358.8457  l:355.4023, f:  0.1785, a:  0.0138,  kl:355.1546  in 2.9s  Heun 82.49% Single 82.66% | heun:0.72s single:0.02s 
Epoch    6/1000 done | avg_loss 350.8863  l:347.0581, f:  0.1723, a:  0.0126,  kl:346.8231  in 2.9s  
Epoch    7/1000 done | avg_loss 343.1284  l:339.5729, f:  0.1595, a:  0.0133,  kl:339.3467  in 2.9s  
Epoch    8/1000 done | avg_loss 335.5671  l:332.2274, f:  0.1704, a:  0.0150,  kl:331.9818  in 2.8s  
Epoch    9/1000 done | avg_loss 328.1439  l:325.0498, f:  0.1550, a:  0.0107,  kl:324.8412  in 2.9s  
Epoch   10/1000 done | avg_loss 320.9053  l:317.5591, f:  0.1498, a:  0.0110,  kl:317.3545  in 2.9s  Heun 83.38% Single 83.72% | heun:0.72s single:0.02s 
Epoch   11/1000 done | avg_loss 315.4864  l:313.7506, f:  0.1470, a:  0.0092,  kl:313.5578  in 2.9s  
Epoch   12/1000 done | avg_loss 311.9914  l:310.7446, f:  0.1399, a:  0.0083,  kl:310.5631  in 2.9s  
Epoch   13/1000 done | avg_loss 308.5653  l:306.6159, f:  0.1355, a:  0.0108,  kl:306.4267  in 2.9s  
Epoch   14/1000 done | avg_loss 305.1285  l:303.8501, f:  0.1279, a:  0.0099,  kl:303.6726  in 2.9s  
Epoch   15/1000 done | avg_loss 301.7420  l:300.4958, f:  0.1266, a:  0.0085,  kl:300.3268  in 2.9s  Heun 86.68% Single 86.82% | heun:0.72s single:0.02s 
Epoch   16/1000 done | avg_loss 298.4267  l:296.9001, f:  0.1255, a:  0.0077,  kl:296.7362  in 2.9s  
Epoch   17/1000 done | avg_loss 295.1280  l:293.5609, f:  0.1187, a:  0.0100,  kl:293.3921  in 2.9s  
Epoch   18/1000 done | avg_loss 291.8371  l:290.1785, f:  0.1146, a:  0.0081,  kl:290.0233  in 2.9s  
Epoch   19/1000 done | avg_loss 288.5791  l:287.0542, f:  0.1165, a:  0.0085,  kl:286.8952  in 2.9s  
Epoch   20/1000 done | avg_loss 285.3890  l:283.8674, f:  0.1190, a:  0.0080,  kl:283.7085  in 3.0s  Heun 87.12% Single 86.86% | heun:0.72s single:0.02s 
Epoch   21/1000 done | avg_loss 282.9474  l:282.1015, f:  0.1050, a:  0.0077,  kl:281.9579  in 3.0s  
Epoch   22/1000 done | avg_loss 281.3567  l:280.7520, f:  0.1050, a:  0.0073,  kl:280.6103  in 2.9s  
Epoch   23/1000 done | avg_loss 279.7667  l:279.1603, f:  0.1003, a:  0.0062,  kl:279.0289  in 2.9s  
Epoch   24/1000 done | avg_loss 278.1985  l:277.6607, f:  0.0927, a:  0.0048,  kl:277.5441  in 2.9s  
Epoch   25/1000 done | avg_loss 276.6371  l:276.1003, f:  0.0912, a:  0.0052,  kl:275.9832  in 3.0s  Heun 88.23% Single 88.17% | heun:0.72s single:0.02s 
Epoch   26/1000 done | avg_loss 275.0906  l:274.2845, f:  0.0872, a:  0.0077,  kl:274.1586  in 2.9s  
Epoch   27/1000 done | avg_loss 273.5932  l:272.7401, f:  0.2089, a:  0.0030,  kl:272.5161  in 2.9s  
Epoch   28/1000 done | avg_loss 272.0662  l:271.2776, f:  0.1421, a:  0.0024,  kl:271.1235  in 2.9s  
Epoch   29/1000 done | avg_loss 270.4987  l:269.6743, f:  0.1400, a:  0.0013,  kl:269.5277  in 2.9s  
Epoch   30/1000 done | avg_loss 268.9700  l:268.3423, f:  0.1399, a:  0.0012,  kl:268.1964  in 2.9s  Heun 89.73% Single 89.96% | heun:0.72s single:0.02s 
Epoch   31/1000 done | avg_loss 267.8185  l:267.6921, f:  0.1416, a:  0.0009,  kl:267.5459  in 2.9s  
Epoch   32/1000 done | avg_loss 267.0460  l:266.5746, f:  0.1349, a:  0.0010,  kl:266.4347  in 3.0s  
Epoch   33/1000 done | avg_loss 266.2759  l:265.7044, f:  0.1375, a:  0.0009,  kl:265.5625  in 2.9s  
Epoch   34/1000 done | avg_loss 265.5390  l:265.0594, f:  0.1332, a:  0.0006,  kl:264.9232  in 2.9s  
Epoch   35/1000 done | avg_loss 264.7805  l:264.6328, f:  0.1345, a:  0.0004,  kl:264.4962  in 3.0s  Heun 90.00% Single 90.28% | heun:0.72s single:0.02s 
Epoch   36/1000 done | avg_loss 264.0237  l:263.7757, f:  0.1333, a:  0.0003,  kl:263.6410  in 2.8s  
Epoch   37/1000 done | avg_loss 263.2829  l:263.0815, f:  0.1301, a:  0.0002,  kl:262.9504  in 2.8s  
Epoch   38/1000 done | avg_loss 262.5188  l:262.0864, f:  0.1328, a:  0.0002,  kl:261.9524  in 2.9s  
Epoch   39/1000 done | avg_loss 261.7686  l:261.2698, f:  0.1295, a:  0.0003,  kl:261.1389  in 2.9s  
Epoch   40/1000 done | avg_loss 261.0175  l:260.4232, f:  0.1303, a:  0.0004,  kl:260.2911  in 2.9s  Heun 90.44% Single 90.43% | heun:0.72s single:0.02s 
Epoch   41/1000 done | avg_loss 260.4405  l:260.3231, f:  0.1349, a:  0.0004,  kl:260.1862  in 2.9s  
Epoch   42/1000 done | avg_loss 260.0800  l:260.1701, f:  0.1340, a:  0.0003,  kl:260.0345  in 2.9s  
Epoch   43/1000 done | avg_loss 259.7128  l:259.3750, f:  0.1288, a:  0.0001,  kl:259.2455  in 2.9s  
Epoch   44/1000 done | avg_loss 259.3287  l:259.0135, f:  0.1347, a:  0.0002,  kl:258.8780  in 2.9s  
Epoch   45/1000 done | avg_loss 258.9574  l:258.8424, f:  0.1315, a:  0.0001,  kl:258.7105  in 2.9s  Heun 90.59% Single 90.41% | heun:0.72s single:0.02s 
Epoch   46/1000 done | avg_loss 258.5725  l:258.5551, f:  0.1277, a:  0.0002,  kl:258.4265  in 2.9s  
Epoch   47/1000 done | avg_loss 258.1941  l:257.9591, f:  0.1323, a:  0.0003,  kl:257.8251  in 2.9s  
Epoch   48/1000 done | avg_loss 257.8167  l:257.7538, f:  0.1284, a:  0.0002,  kl:257.6243  in 2.9s  
Epoch   49/1000 done | avg_loss 257.4485  l:257.4992, f:  0.1304, a:  0.0001,  kl:257.3682  in 3.0s  
Epoch   50/1000 done | avg_loss 257.0772  l:256.8453, f:  0.1325, a:  0.0001,  kl:256.7123  in 2.9s  Heun 90.52% Single 90.68% | heun:0.72s single:0.02s 
Epoch   51/1000 done | avg_loss 256.7813  l:256.5512, f:  0.1296, a:  0.0001,  kl:256.4212  in 3.0s  
Epoch   52/1000 done | avg_loss 256.6064  l:256.2531, f:  0.1299, a:  0.0001,  kl:256.1226  in 3.0s  
Epoch   53/1000 done | avg_loss 256.4188  l:256.7854, f:  0.1328, a:  0.0001,  kl:256.6520  in 2.9s  
Epoch   54/1000 done | avg_loss 256.2252  l:255.8163, f:  0.1287, a:  0.0001,  kl:255.6869  in 3.0s  
Epoch   55/1000 done | avg_loss 256.0371  l:255.8615, f:  0.1244, a:  0.0001,  kl:255.7366  in 2.9s  Heun 90.34% Single 90.58% | heun:0.72s single:0.02s 
Epoch   56/1000 done | avg_loss 255.8504  l:255.5961, f:  0.1274, a:  0.0001,  kl:255.4682  in 2.9s  
Epoch   57/1000 done | avg_loss 255.6571  l:255.5212, f:  0.1291, a:  0.0001,  kl:255.3917  in 2.9s  
Epoch   58/1000 done | avg_loss 255.4729  l:255.5172, f:  0.1257, a:  0.0001,  kl:255.3911  in 3.0s  
Epoch   59/1000 done | avg_loss 255.2648  l:255.2449, f:  0.1275, a:  0.0001,  kl:255.1171  in 3.0s  
Epoch   60/1000 done | avg_loss 255.1060  l:255.1055, f:  0.1301, a:  0.0001,  kl:254.9750  in 2.9s  Heun 90.29% Single 90.37% | heun:0.72s single:0.02s 
Epoch   61/1000 done | avg_loss 254.9597  l:255.3210, f:  0.1293, a:  0.0001,  kl:255.1914  in 2.9s  
Epoch   62/1000 done | avg_loss 254.8499  l:254.8124, f:  0.1274, a:  0.0001,  kl:254.6845  in 2.9s  
Epoch   63/1000 done | avg_loss 254.7708  l:254.9742, f:  0.1305, a:  0.0001,  kl:254.8433  in 2.9s  
Epoch   64/1000 done | avg_loss 254.6621  l:254.6749, f:  0.1324, a:  0.0001,  kl:254.5421  in 2.9s  
Epoch   65/1000 done | avg_loss 254.5839  l:254.6772, f:  0.1282, a:  0.0001,  kl:254.5486  in 3.0s  Heun 90.45% Single 90.36% | heun:0.72s single:0.02s 
Epoch   66/1000 done | avg_loss 254.4856  l:254.3813, f:  0.1342, a:  0.0001,  kl:254.2467  in 2.9s  
Epoch   67/1000 done | avg_loss 254.3729  l:254.3380, f:  0.1308, a:  0.0001,  kl:254.2068  in 2.9s  
Epoch   68/1000 done | avg_loss 254.2973  l:254.2297, f:  0.1284, a:  0.0000,  kl:254.1011  in 2.9s  
Epoch   69/1000 done | avg_loss 254.1920  l:254.1957, f:  0.1283, a:  0.0001,  kl:254.0670  in 2.9s  
Epoch   70/1000 done | avg_loss 254.1086  l:253.9858, f:  0.1298, a:  0.0001,  kl:253.8557  in 2.9s  Heun 90.53% Single 90.42% | heun:0.73s single:0.02s 
Epoch   71/1000 done | avg_loss 254.0541  l:254.2238, f:  0.1276, a:  0.0001,  kl:254.0959  in 3.0s  
Epoch   72/1000 done | avg_loss 253.9878  l:253.8124, f:  0.1328, a:  0.0001,  kl:253.6793  in 2.9s  
Epoch   73/1000 done | avg_loss 253.9545  l:253.7978, f:  0.1300, a:  0.0000,  kl:253.6675  in 2.9s  
Epoch   74/1000 done | avg_loss 253.8898  l:253.6113, f:  0.1299, a:  0.0001,  kl:253.4812  in 2.9s  
Epoch   75/1000 done | avg_loss 253.8561  l:254.0869, f:  0.1319, a:  0.0000,  kl:253.9547  in 2.9s  Heun 90.51% Single 90.27% | heun:0.72s single:0.02s 
Epoch   76/1000 done | avg_loss 253.8118  l:253.5912, f:  0.1334, a:  0.0001,  kl:253.4575  in 3.0s  
Epoch   77/1000 done | avg_loss 253.7552  l:253.8554, f:  0.1316, a:  0.0000,  kl:253.7236  in 3.0s  
Epoch   78/1000 done | avg_loss 253.7128  l:253.4964, f:  0.1275, a:  0.0001,  kl:253.3686  in 2.9s  
Epoch   79/1000 done | avg_loss 253.6543  l:253.5521, f:  0.1276, a:  0.0001,  kl:253.4243  in 2.9s  
Epoch   80/1000 done | avg_loss 253.6034  l:253.4901, f:  0.1240, a:  0.0000,  kl:253.3659  in 3.0s  Heun 90.70% Single 90.48% | heun:0.72s single:0.02s 
Epoch   81/1000 done | avg_loss 253.5491  l:253.4218, f:  0.1277, a:  0.0001,  kl:253.2938  in 2.9s  
Epoch   82/1000 done | avg_loss 253.5524  l:253.7048, f:  0.1319, a:  0.0001,  kl:253.5726  in 2.9s  
Epoch   83/1000 done | avg_loss 253.5340  l:253.8168, f:  0.1312, a:  0.0001,  kl:253.6854  in 2.9s  
Epoch   84/1000 done | avg_loss 253.4999  l:253.4954, f:  0.1254, a:  0.0000,  kl:253.3698  in 3.0s  
Epoch   85/1000 done | avg_loss 253.4737  l:253.5495, f:  0.1335, a:  0.0000,  kl:253.4158  in 2.9s  Heun 90.65% Single 90.44% | heun:0.72s single:0.02s 
Epoch   86/1000 done | avg_loss 253.4539  l:253.1737, f:  0.1281, a:  0.0000,  kl:253.0453  in 3.0s  
Epoch   87/1000 done | avg_loss 253.4181  l:253.5672, f:  0.1301, a:  0.0000,  kl:253.4368  in 3.0s  
Epoch   88/1000 done | avg_loss 253.3889  l:253.4088, f:  0.1286, a:  0.0000,  kl:253.2800  in 3.0s  
Epoch   89/1000 done | avg_loss 253.3953  l:253.4645, f:  0.1305, a:  0.0000,  kl:253.3338  in 3.0s  
Epoch   90/1000 done | avg_loss 253.3554  l:253.6349, f:  0.1317, a:  0.0001,  kl:253.5030  in 2.9s  Heun 90.35% Single 90.45% | heun:0.72s single:0.02s 
Epoch   91/1000 done | avg_loss 253.3443  l:253.4743, f:  0.1296, a:  0.0001,  kl:253.3444  in 2.9s  
Epoch   92/1000 done | avg_loss 253.3307  l:253.3333, f:  0.1314, a:  0.0001,  kl:253.2016  in 2.9s  
Epoch   93/1000 done | avg_loss 253.3202  l:253.2735, f:  0.1319, a:  0.0000,  kl:253.1414  in 2.9s  
Epoch   94/1000 done | avg_loss 253.2898  l:253.3094, f:  0.1329, a:  0.0001,  kl:253.1761  in 2.9s  
Epoch   95/1000 done | avg_loss 253.2953  l:253.1851, f:  0.1299, a:  0.0000,  kl:253.0550  in 2.9s  Heun 90.27% Single 90.35% | heun:0.72s single:0.02s 
Epoch   96/1000 done | avg_loss 253.2674  l:253.2179, f:  0.1367, a:  0.0000,  kl:253.0811  in 3.0s  
Epoch   97/1000 done | avg_loss 253.2695  l:253.2925, f:  0.1318, a:  0.0000,  kl:253.1605  in 3.0s  
Epoch   98/1000 done | avg_loss 253.2628  l:253.2405, f:  0.1312, a:  0.0000,  kl:253.1091  in 2.9s  
Epoch   99/1000 done | avg_loss 253.2464  l:252.9178, f:  0.1341, a:  0.0001,  kl:252.7834  in 3.1s  
Epoch  100/1000 done | avg_loss 253.2381  l:253.1085, f:  0.1312, a:  0.0000,  kl:252.9771  in 3.0s  Heun 90.59% Single 90.47% | heun:0.72s single:0.02s 
Epoch  101/1000 done | avg_loss 253.2387  l:253.3015, f:  0.1311, a:  0.0000,  kl:253.1703  in 3.1s  
Epoch  102/1000 done | avg_loss 253.2263  l:253.4427, f:  0.1343, a:  0.0000,  kl:253.3082  in 2.9s  
Epoch  103/1000 done | avg_loss 253.2059  l:253.1668, f:  0.1325, a:  0.0001,  kl:253.0339  in 2.9s  
Epoch  104/1000 done | avg_loss 253.2026  l:253.1765, f:  0.1318, a:  0.0000,  kl:253.0446  in 2.9s  
Epoch  105/1000 done | avg_loss 253.2079  l:253.0677, f:  0.1314, a:  0.0000,  kl:252.9361  in 2.9s  Heun 90.53% Single 90.34% | heun:0.72s single:0.02s 
Epoch  106/1000 done | avg_loss 253.1934  l:252.9671, f:  0.1349, a:  0.0000,  kl:252.8321  in 2.9s  
Epoch  107/1000 done | avg_loss 253.2088  l:253.1227, f:  0.1347, a:  0.0000,  kl:252.9878  in 3.0s  
Epoch  108/1000 done | avg_loss 253.1820  l:253.1320, f:  0.1335, a:  0.0000,  kl:252.9984  in 3.0s  
Epoch  109/1000 done | avg_loss 253.1691  l:253.1788, f:  0.1297, a:  0.0000,  kl:253.0490  in 2.9s  
Epoch  110/1000 done | avg_loss 253.1626  l:253.1770, f:  0.1341, a:  0.0000,  kl:253.0427  in 3.4s  Heun 90.50% Single 90.51% | heun:0.72s single:0.02s 
Epoch  111/1000 done | avg_loss 253.1670  l:253.1411, f:  0.1345, a:  0.0000,  kl:253.0063  in 3.0s  
Epoch  112/1000 done | avg_loss 253.1806  l:253.1036, f:  0.1335, a:  0.0000,  kl:252.9699  in 3.0s  
Epoch  113/1000 done | avg_loss 253.1523  l:253.1476, f:  0.1366, a:  0.0000,  kl:253.0109  in 3.0s  
Epoch  114/1000 done | avg_loss 253.1629  l:253.4103, f:  0.1352, a:  0.0000,  kl:253.2749  in 3.0s  
Epoch  115/1000 done | avg_loss 253.1632  l:253.3116, f:  0.1364, a:  0.0000,  kl:253.1750  in 3.0s  Heun 90.52% Single 90.51% | heun:0.72s single:0.02s 
Epoch  116/1000 done | avg_loss 253.1615  l:253.1789, f:  0.1302, a:  0.0000,  kl:253.0486  in 3.0s  
Epoch  117/1000 done | avg_loss 253.1661  l:253.0899, f:  0.1344, a:  0.0000,  kl:252.9554  in 3.0s  
Epoch  118/1000 done | avg_loss 253.1600  l:253.1501, f:  0.1373, a:  0.0000,  kl:253.0125  in 2.9s  
Epoch  119/1000 done | avg_loss 253.1420  l:252.9675, f:  0.1332, a:  0.0000,  kl:252.8342  in 2.9s  
Epoch  120/1000 done | avg_loss 253.1470  l:253.3162, f:  0.1348, a:  0.0000,  kl:253.1813  in 3.0s  Heun 90.36% Single 90.59% | heun:0.72s single:0.02s 
Epoch  121/1000 done | avg_loss 253.1452  l:253.1548, f:  0.1370, a:  0.0000,  kl:253.0177  in 3.0s  
Epoch  122/1000 done | avg_loss 253.1317  l:252.9985, f:  0.1384, a:  0.0000,  kl:252.8599  in 2.9s  
Epoch  123/1000 done | avg_loss 253.1563  l:253.0627, f:  0.1387, a:  0.0000,  kl:252.9239  in 2.9s  
Epoch  124/1000 done | avg_loss 253.1387  l:253.1695, f:  0.1365, a:  0.0000,  kl:253.0328  in 3.0s  
Epoch  125/1000 done | avg_loss 253.1335  l:253.1572, f:  0.1313, a:  0.0000,  kl:253.0257  in 3.1s  Heun 90.55% Single 90.44% | heun:0.72s single:0.02s 
Epoch  126/1000 done | avg_loss 253.1293  l:253.3053, f:  0.1371, a:  0.0000,  kl:253.1681  in 3.0s  
Epoch  127/1000 done | avg_loss 253.1441  l:253.2950, f:  0.1374, a:  0.0000,  kl:253.1575  in 3.0s  
Epoch  128/1000 done | avg_loss 253.1233  l:253.0338, f:  0.1357, a:  0.0000,  kl:252.8979  in 3.0s  
Epoch  129/1000 done | avg_loss 253.1357  l:252.9799, f:  0.1375, a:  0.0000,  kl:252.8422  in 3.1s  
Epoch  130/1000 done | avg_loss 253.1234  l:253.1888, f:  0.1396, a:  0.0000,  kl:253.0490  in 3.0s  Heun 90.60% Single 90.33% | heun:0.72s single:0.02s 
Epoch  131/1000 done | avg_loss 253.1273  l:253.6703, f:  0.1367, a:  0.0000,  kl:253.5334  in 2.9s  
Epoch  132/1000 done | avg_loss 253.1230  l:253.3374, f:  0.1353, a:  0.0000,  kl:253.2020  in 2.9s  
Epoch  133/1000 done | avg_loss 253.1135  l:253.2011, f:  0.1346, a:  0.0000,  kl:253.0663  in 3.0s  
Epoch  134/1000 done | avg_loss 253.1301  l:253.0278, f:  0.1360, a:  0.0001,  kl:252.8916  in 3.0s  
Epoch  135/1000 done | avg_loss 253.1231  l:253.3562, f:  0.1380, a:  0.0000,  kl:253.2180  in 2.9s  Heun 90.39% Single 90.56% | heun:0.72s single:0.02s 
Epoch  136/1000 done | avg_loss 253.1177  l:252.8772, f:  0.1357, a:  0.0000,  kl:252.7415  in 2.9s  
Epoch  137/1000 done | avg_loss 253.1337  l:253.2292, f:  0.1426, a:  0.0000,  kl:253.0864  in 3.0s  
Epoch  138/1000 done | avg_loss 253.1205  l:253.0862, f:  0.1407, a:  0.0000,  kl:252.9454  in 3.0s  
Epoch  139/1000 done | avg_loss 253.1254  l:253.0942, f:  0.1400, a:  0.0000,  kl:252.9541  in 3.0s  
Epoch  140/1000 done | avg_loss 253.1208  l:253.2209, f:  0.1404, a:  0.0000,  kl:253.0804  in 3.0s  Heun 90.47% Single 90.56% | heun:0.72s single:0.02s 
Epoch  141/1000 done | avg_loss 253.1227  l:253.2420, f:  0.1382, a:  0.0000,  kl:253.1036  in 3.0s  
Epoch  142/1000 done | avg_loss 253.1371  l:253.0499, f:  0.1411, a:  0.0000,  kl:252.9087  in 3.0s  
Epoch  143/1000 done | avg_loss 253.1248  l:253.2739, f:  0.1393, a:  0.0000,  kl:253.1346  in 3.1s  
Epoch  144/1000 done | avg_loss 253.1157  l:253.2217, f:  0.1369, a:  0.0000,  kl:253.0847  in 3.0s  
Epoch  145/1000 done | avg_loss 253.1211  l:253.2343, f:  0.1403, a:  0.0000,  kl:253.0938  in 3.0s  Heun 90.50% Single 90.36% | heun:0.72s single:0.02s 
Epoch  146/1000 done | avg_loss 253.1219  l:253.0769, f:  0.1395, a:  0.0000,  kl:252.9373  in 3.0s  
Epoch  147/1000 done | avg_loss 253.1215  l:253.2811, f:  0.1405, a:  0.0000,  kl:253.1405  in 2.9s  
Epoch  148/1000 done | avg_loss 253.1284  l:253.0107, f:  0.1422, a:  0.0000,  kl:252.8683  in 2.9s  
Epoch  149/1000 done | avg_loss 253.1301  l:252.8881, f:  0.1397, a:  0.0000,  kl:252.7483  in 3.0s  
Epoch  150/1000 done | avg_loss 253.1217  l:253.1141, f:  0.1427, a:  0.0000,  kl:252.9713  in 3.0s  Heun 90.48% Single 90.29% | heun:0.72s single:0.02s 
Epoch  151/1000 done | avg_loss 253.1116  l:253.0898, f:  0.1404, a:  0.0000,  kl:252.9492  in 3.0s  
Epoch  152/1000 done | avg_loss 253.1127  l:253.0184, f:  0.1412, a:  0.0000,  kl:252.8771  in 3.1s  
Epoch  153/1000 done | avg_loss 253.1244  l:253.4260, f:  0.1449, a:  0.0000,  kl:253.2811  in 3.1s  
Epoch  154/1000 done | avg_loss 253.1372  l:253.2727, f:  0.1401, a:  0.0000,  kl:253.1324  in 3.1s  
Epoch  155/1000 done | avg_loss 253.1058  l:253.4835, f:  0.1422, a:  0.0000,  kl:253.3412  in 3.1s  Heun 90.44% Single 90.54% | heun:0.72s single:0.02s 
Epoch  156/1000 done | avg_loss 253.1229  l:253.0665, f:  0.1471, a:  0.0000,  kl:252.9193  in 3.0s  
Epoch  157/1000 done | avg_loss 253.1369  l:252.9559, f:  0.1397, a:  0.0000,  kl:252.8161  in 3.1s  
Epoch  158/1000 done | avg_loss 253.1219  l:253.0016, f:  0.1448, a:  0.0000,  kl:252.8566  in 3.0s  
Epoch  159/1000 done | avg_loss 253.1218  l:252.9831, f:  0.1454, a:  0.0000,  kl:252.8376  in 3.0s  
Epoch  160/1000 done | avg_loss 253.1190  l:252.9512, f:  0.1452, a:  0.0000,  kl:252.8059  in 3.1s  Heun 90.40% Single 90.61% | heun:0.72s single:0.02s 
Epoch  161/1000 done | avg_loss 253.1123  l:253.0910, f:  0.1391, a:  0.0000,  kl:252.9519  in 3.0s  
Epoch  162/1000 done | avg_loss 253.1174  l:253.0383, f:  0.1410, a:  0.0000,  kl:252.8972  in 3.1s  
Epoch  163/1000 done | avg_loss 253.1218  l:253.2365, f:  0.1429, a:  0.0000,  kl:253.0934  in 3.0s  
Epoch  164/1000 done | avg_loss 253.1234  l:252.9613, f:  0.1421, a:  0.0000,  kl:252.8192  in 3.0s  
Epoch  165/1000 done | avg_loss 253.1099  l:252.9915, f:  0.1443, a:  0.0000,  kl:252.8471  in 3.1s  Heun 90.47% Single 90.48% | heun:0.72s single:0.02s 
Epoch  166/1000 done | avg_loss 253.1265  l:252.9957, f:  0.1398, a:  0.0000,  kl:252.8558  in 3.1s  
Epoch  167/1000 done | avg_loss 253.1229  l:253.1965, f:  0.1418, a:  0.0000,  kl:253.0546  in 3.0s  
Epoch  168/1000 done | avg_loss 253.1105  l:252.8992, f:  0.1414, a:  0.0000,  kl:252.7577  in 3.0s  
Epoch  169/1000 done | avg_loss 253.1245  l:252.9682, f:  0.1417, a:  0.0000,  kl:252.8264  in 3.0s  
Epoch  170/1000 done | avg_loss 253.1177  l:253.5296, f:  0.1411, a:  0.0000,  kl:253.3885  in 3.0s  Heun 90.51% Single 90.67% | heun:0.72s single:0.02s 
Epoch  171/1000 done | avg_loss 253.1265  l:253.0833, f:  0.1425, a:  0.0000,  kl:252.9408  in 3.0s  
Epoch  172/1000 done | avg_loss 253.1275  l:252.9296, f:  0.1459, a:  0.0000,  kl:252.7836  in 3.0s  
Epoch  173/1000 done | avg_loss 253.1286  l:253.2162, f:  0.1452, a:  0.0000,  kl:253.0708  in 3.0s  
Epoch  174/1000 done | avg_loss 253.1211  l:253.0502, f:  0.1435, a:  0.0000,  kl:252.9066  in 3.1s  
Epoch  175/1000 done | avg_loss 253.1313  l:252.9746, f:  0.1450, a:  0.0000,  kl:252.8294  in 3.0s  Heun 90.58% Single 90.42% | heun:0.72s single:0.02s 
```
