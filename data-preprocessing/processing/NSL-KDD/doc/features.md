# NSL-KDD

Disclaimer: This is a simple listing of important features. Please refer to `util/documentation.pdf` for detailed information.

## Binary:

`Target Columns`: anomaly_bool

`Key Features`:<br>
- duration
- protocol_type
- service
- src_bytes/dst_bytes
- count
- srv_count
- serror_rate
- srv_serror_rate
- same_srv_rate

`Goal`: Classify, wheter normal traffic or attack.

`Focus on`: abnormal volume, host/service interaction.

=== Positive correlation with 'anomaly_bool' ===

1.  dst_host_srv_serror_rate :0.615
2.  dst_host_serror_rate     :0.613
3.  serror_rate              :0.611
4.  srv_serror_rate          :0.609
5.  count                    :0.551
6.  dst_host_count           :0.375
7.  dst_host_rerror_rate     :0.294
8.  rerror_rate              :0.293
9.  srv_rerror_rate          :0.292
10. dst_host_srv_rerror_rate :0.292
11. service                  :0.291
12. dst_host_diff_srv_rate   :0.250
13. diff_srv_rate            :0.212
14. dst_host_same_src_port_rate :0.087
15. wrong_fragment           :0.085
16. dst_host_srv_diff_host_rate :0.054
17. num_failed_logins        :0.045
18. duration                 :0.045
19. land                     :0.008
20. is_guest_login           :0.007
21. src_bytes                :0.006
22. is_host_login            :0.004
23. dst_bytes                :0.004


## Multi-class:

`Target Column`: attack_number

`Key Features`:<br>
- logged_in
- num_failed_logins
- hot
- num_root
- root_shell
- dst_host_* rates

`Goal`: Classify the detailed attack type (e.g., DoS, bruteforce, etc.)

`Focus on`: login misuse, privilege escalation, scan diversity

`Note`: The dataset is still imbalanced regarding the multiclass target column. That should be adjusted before the multiclass model training. Since the main goal of this project is a binary classifier, we will skip this step.


=== Positive correlation with 'attack_number' ===

1.  same_srv_rate            :0.595
2.  dst_host_same_srv_rate   :0.429
3.  flag                     :0.421
4.  dst_host_same_src_port_rate :0.390
5.  logged_in                :0.332
6.  attack_label             :0.331
7.  dst_host_srv_diff_host_rate :0.318
8.  dst_host_srv_count       :0.302
9.  dst_host_diff_srv_rate   :0.264
10. srv_diff_host_rate       :0.248
11. is_guest_login           :0.198
12. num_failed_logins        :0.165
13. diff_srv_rate            :0.139
14. duration                 :0.138
15. hot                      :0.112
16. root_shell               :0.051
17. protocol_type            :0.048
18. num_shells               :0.032
19. num_access_files         :0.018
20. is_host_login            :0.017
21. urgent                   :0.015
22. num_file_creations       :0.015
23. src_bytes                :0.012
24. su_attempted             :0.008
25. dst_bytes                :0.007
26. num_root                 :0.004
27. num_compromised          :0.004

