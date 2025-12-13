# CICIDS2017 – Key Features

Disclaimer: This is a simple listing of important features. Please refer to `util/documentation.pdf` for detailed information.

## Binary Classification:

`Target Column`: anomaly_bool

`Key Features`:<br>
- Flow Duration
- Flow Bytes/s
- Flow Packets/s
- Fwd/Bwd Packet Length Mean/Std
- Flow IAT Mean/Std
- SYN/ACK/PSH flag counts

`Goal`: Classify, wheter normal traffic or attack.

`Focus on`: global traffic anomalies


=== Positive correlation with 'anomaly_bool' ===

1.  Attack Number            :0.932
2.  Bwd Packet Length Std    :0.612
3.  Bwd Packet Length Max    :0.596
4.  Bwd Packet Length Mean   :0.594
5.  Avg Bwd Segment Size     :0.594
6.  Packet Length Std        :0.573
7.  Max Packet Length        :0.555
8.  Packet Length Variance   :0.537
9.  Average Packet Size      :0.522
10. Packet Length Mean       :0.521
11. Fwd IAT Std              :0.506
12. Idle Max                 :0.476
13. Flow IAT Max             :0.471
14. Idle Mean                :0.471
15. Fwd IAT Max              :0.471
16. Idle Min                 :0.458
17. Flow IAT Std             :0.411
18. Fwd IAT Total            :0.280
19. Flow Duration            :0.279
20. FIN Flag Count           :0.271
21. Flow IAT Mean            :0.225
22. Bwd IAT Std              :0.198
23. Fwd IAT Mean             :0.193
24. Bwd IAT Max              :0.146
25. PSH Flag Count           :0.145
26. Idle Std                 :0.129
27. ACK Flag Count           :0.103
28. Active Min               :0.037
29. Active Mean              :0.028
30. Bwd IAT Mean             :0.022
31. Flow IAT Min             :0.018
32. Bwd Packets/s            :0.016
33. Bwd IAT Total            :0.006
34. min_seg_size_forward     :0.001
35. Bwd Header Length        :0.001
36. Fwd Header Length        :0.001
37. Fwd Header Length.1      :0.001





## Multi-class Classification:

`Target Columns`: Attack Number

`Key Features`:<br>
- Fwd/Bwd IAT max
- Active/Idle Mean/Std
- SYN/FIN/RST Flag counts

`Goal`: Classify the detailed attack type (e.g., DoS, bruteforce, etc.)

`Focus on`: login misuse, privilege escalation, scan diversity

`Note`: The dataset is still imbalanced regarding the multiclass target column. That should be adjusted before the multiclass model training. Since the main goal of this project is a binary classifier, we will skip this step.

=== Positive correlation with 'Attack Number' ===

1.  anomaly_bool             :0.932
2.  Bwd Packet Length Std    :0.447
3.  Bwd Packet Length Max    :0.436
4.  Bwd Packet Length Mean   :0.431
5.  Avg Bwd Segment Size     :0.431
6.  Packet Length Std        :0.412
7.  Fwd IAT Std              :0.410
8.  Max Packet Length        :0.402
9.  Idle Max                 :0.384
10. Packet Length Variance   :0.384
11. Idle Mean                :0.383
12. Fwd IAT Max              :0.379
13. Flow IAT Max             :0.379
14. Idle Min                 :0.376
15. Packet Length Mean       :0.366
16. Average Packet Size      :0.364
17. Flow IAT Std             :0.328
18. FIN Flag Count           :0.230
19. Fwd IAT Total            :0.215
20. Flow Duration            :0.213
21. PSH Flag Count           :0.209
22. Flow IAT Mean            :0.175
23. Bwd IAT Std              :0.163
24. Fwd IAT Mean             :0.152
25. Bwd IAT Max              :0.118
26. Idle Std                 :0.081
27. Bwd Packets/s            :0.067
28. Init_Win_bytes_forward   :0.037
29. ACK Flag Count           :0.032
30. Active Min               :0.020
31. Flow IAT Min             :0.014
32. Bwd IAT Mean             :0.013
33. Active Mean              :0.012
34. min_seg_size_forward     :0.001
35. Bwd Header Length        :0.001
36. Fwd Header Length        :0.001
37. Fwd Header Length.1      :0.001





