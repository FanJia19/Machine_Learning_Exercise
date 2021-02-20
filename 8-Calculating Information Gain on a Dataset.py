def two_group_ent(first, tot):                        
    return -(first/tot*np.log2(first/tot) +           
             (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)  # 先计算两种虫子M&L的 Entropy                   
g17_ent = 15/24 * two_group_ent(11,15) +  # 算>17的虫子的Entropy
           9/24 * two_group_ent(6,9)      # 算<17的虫子的Entropy           

answer = tot_ent - g17_ent   