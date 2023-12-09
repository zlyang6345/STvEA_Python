corrected_codex_protein
            B220  CD106  CD11b  CD11c  ...  MHCII  NKp46  TCR  Ter119
codex_1231    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_1316    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_1750    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_2107    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_4022    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_11477   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12268   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12346   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12452   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12722   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_13748   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_13806   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_22880   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_25519   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN

bv.loc[na_index]
Out[12]: 
             B220  CD106  CD11b  CD11c  ...  MHCII  NKp46  TCR  Ter119
codex_1231    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_1316    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_1750    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_2107    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_4022    NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_11477   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12268   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12346   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12452   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_12722   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_13748   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_13806   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_22880   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN
codex_25519   NaN    NaN    NaN    NaN  ...    NaN    NaN  NaN     NaN

weights.loc[na_index]
Out[13]: 
             0      1      2      3      ...  32064  32065  32066  32067
codex_1231     NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_1316     NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_1750     NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_2107     NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_4022     NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_11477    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_12268    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_12346    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_12452    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_12722    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_13748    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_13806    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_22880    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN
codex_25519    NaN    NaN    NaN    NaN  ...    NaN    NaN    NaN    NaN

query_mat.loc[na_rows, :]
Out[7]: 
            B220  CD106  CD11b  CD11c  CD16.32     CD169  CD19  CD21.35  \
codex_1231  -10.0  -10.0  -10.0  -10.0    -10.0 -0.504489 -10.0    -10.0   
codex_1316  -10.0  -10.0  -10.0  -10.0    -10.0 -0.457076 -10.0    -10.0   
codex_1750  -10.0  -10.0  -10.0  -10.0    -10.0 -0.332171 -10.0    -10.0   
codex_2107  -10.0  -10.0  -10.0  -10.0    -10.0 -0.574714 -10.0    -10.0   
codex_4022  -10.0  -10.0  -10.0  -10.0    -10.0  0.772898 -10.0    -10.0   
codex_11477 -10.0  -10.0  -10.0  -10.0    -10.0 -0.124009 -10.0    -10.0   
codex_12268 -10.0  -10.0  -10.0  -10.0    -10.0 -0.618761 -10.0    -10.0   
codex_12346 -10.0  -10.0  -10.0  -10.0    -10.0 -0.071585 -10.0    -10.0   
codex_12452 -10.0  -10.0  -10.0  -10.0    -10.0 -0.569053 -10.0    -10.0   
codex_12722 -10.0  -10.0  -10.0  -10.0    -10.0  0.833520 -10.0    -10.0   
codex_13748 -10.0  -10.0  -10.0  -10.0    -10.0  0.227689 -10.0    -10.0   
codex_13806 -10.0  -10.0  -10.0  -10.0    -10.0 -0.555804 -10.0    -10.0   
codex_22880 -10.0  -10.0  -10.0  -10.0    -10.0 -0.612228 -10.0    -10.0   
codex_25519 -10.0  -10.0  -10.0  -10.0    -10.0 -0.217049 -10.0    -10.0   
             CD27   CD3  CD31  CD35   CD4  CD44  CD45   CD5  CD71  CD79b  \
codex_1231  -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_1316  -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_1750  -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_2107  -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_4022  -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_11477 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_12268 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_12346 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_12452 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_12722 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_13748 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_13806 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_22880 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
codex_25519 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0  -10.0   
             CD8a  CD90  ERTR7  F4.80   IgD   IgM  Ly6C  Ly6G  MHCII  NKp46  \
codex_1231  -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_1316  -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_1750  -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_2107  -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_4022  -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_11477 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_12268 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_12346 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_12452 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_12722 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_13748 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_13806 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_22880 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
codex_25519 -10.0 -10.0  -10.0  -10.0 -10.0 -10.0 -10.0 -10.0  -10.0  -10.0   
              TCR  Ter119  
codex_1231  -10.0   -10.0  
codex_1316  -10.0   -10.0  
codex_1750  -10.0   -10.0  
codex_2107  -10.0   -10.0  
codex_4022  -10.0   -10.0  
codex_11477 -10.0   -10.0  
codex_12268 -10.0   -10.0  
codex_12346 -10.0   -10.0  
codex_12452 -10.0   -10.0  
codex_12722 -10.0   -10.0  
codex_13748 -10.0   -10.0  
codex_13806 -10.0   -10.0  
codex_22880 -10.0   -10.0  
codex_25519 -10.0   -10.0  

[14 rows x 30 columns]