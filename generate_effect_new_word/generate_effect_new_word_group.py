def concat_prop_words(cut_iter,text,text_id):
    #1 由于python2.7 的默认encoding是 ascii , 与unicode存在兼容问题，遂需要设置一下。 但 py3不需要进行这个设置
    #reload(sys) ;    sys.setdefaultencoding('utf-8')
    #2 组合落单词，使为有意义新词
    word_flag=[[word,flag] for word,flag in cut_iter]  
    #print(word_flag)
    # 2.1 基于词性的粘滞力权重，粘滞力分为向上粘滞和向下粘滞，这里默认给出向下粘滞力权重，则向上粘滞力=10-向下粘滞力权重
    prof_weight={'a':5.2,'n':3.5,'d':6.5,'v':6.5,'vn':3.5,'ns':3.5,'nt':3.5,'nz':3.5,'an':3.5,'x':0,'q':2,'uj':4,'ul':4,'m':3,'r':6} 
    stop_words_list=[u'是',u'也',u'着',u'很',u'到',u'就',u'刚',u'才',u'蛮',u'都',u'让',u'太',u'去',u'来',u'找',u'上',u'下',u'卖',u'说',u'有',u'叫',u'起',u'要']
    special_words_list=[u'不',u'没',u'无']
    wlen=len(word_flag)
    rst=[]
    if wlen>0:
        rst.append(word_flag[0])
    loci=1;not_concat=1
    while loci<wlen:
        wf_loc=word_flag[loci]
        wf_pre=word_flag[loci-1]
        try:
            wf_suf=word_flag[loci+1]
        except:
            wf_suf=['。','x']
        if len(wf_loc[0])==1 and wf_loc[0] not in stop_words_list and wf_loc[1] not in 'ujulxepcykzgqrmd' and wf_loc[0] not in special_words_list :
            # 2.3 基于词性判断，获得计算时所需的权重值
            pre_i_w=1;suf_i_w=1;loc_i_w=1
            if wf_loc[1] in prof_weight.keys():
                loc_i_w=prof_weight.get(wf_loc[1])
            if wf_pre[1] in prof_weight.keys():
                pre_i_w=prof_weight.get(wf_pre[1])
            if wf_suf[1] in prof_weight.keys():
                suf_i_w=prof_weight.get(wf_suf[1])
            if loc_i_w in [0,1]:  
                inverse_loc_i_w=loc_i_w
            else:
                inverse_loc_i_w=10-loc_i_w
            if suf_i_w in [0,1]:  
                inverse_suf_i_w=suf_i_w
            else:
                inverse_suf_i_w=10-suf_i_w
                
            # 2.4 计算粘滞方向，并做判断 ，进行连接
            #if pre_i_w*pre_i_w*inverse_loc_i_w/math.pow(len(wf_pre[0]),2)*not_concat>=loc_i_w*loc_i_w*inverse_suf_i_w/math.pow(len(wf_suf[0]),2) :
            #print(wf_loc[0],pre_i_w*pre_i_w*inverse_loc_i_w/len(wf_pre[0])*not_concat,loc_i_w*loc_i_w*inverse_suf_i_w/len(wf_suf[0]))
            pre=pre_i_w*pre_i_w*inverse_loc_i_w/math.ceil(len(wf_pre[0])/2.0)*not_concat  
            suf=loc_i_w*loc_i_w*inverse_suf_i_w/math.ceil(len(wf_suf[0])/2.0)  
            if pre>=suf and pre>0:
                if wf_pre[1]!=wf_loc[1]:
                    del rst[-1]  # 先删掉上一个
                    rst.append([wf_pre[0]+wf_loc[0],wf_pre[1]+wf_loc[1]]) # 把连接的存入result集合
                else:
                    rst.append(wf_loc)
                loci=loci+1
                not_concat=1
            elif pre<suf and suf>0:
                if wf_suf[1]!=wf_loc[1]:
                    rst.append([wf_loc[0]+wf_suf[0],wf_loc[1]+wf_suf[1]])
                    if len(wf_suf[0])>1 :  
                        loci=loci+2
                    else:
                        loci=loci+1
                else:
                    rst.append(wf_loc)
                    loci=loci+1
                not_concat=0.2
            else:
                rst.append(wf_loc)
                loci=loci+1
                not_concat=1
        elif wf_loc[0] in special_words_list and wf_suf[1]!='x':
            rst.append([wf_loc[0]+wf_suf[0],wf_loc[1]+wf_suf[1]])
            if len(wf_suf[0])>1 :
                loci=loci+2
            else:
                loci=loci+1
            not_concat=0.2
        else:
            rst.append(wf_loc)
            loci=loci+1
            not_concat=1
    
    # 3 输出有效词 作为结果 
    rst=[(text_id,text,[x[0] for x in rst if len(x[0])>1 and set(x[1]).intersection(set('anvd'))!=set() and x[1]!='eng'])] # 面向element的flatmap  
    #print(rst)
    return rst
 
