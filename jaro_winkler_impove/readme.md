**识别短文本的相似度:**   
jaro-winkler-improving.py   
如品牌名称，这里实现了jaro-winkler算法，并提供了3种优化方案。    
流行方法 ： 1/3×(m/s1+m/s2+(m-t)/m)   #两个字符串中匹配度和匹配后的换位比例 相加，三者的平均值 ： 存在普遍偏高的问题    
改进1   :  1/2×(m/s1+m/s2) × (m-t)/m  #先求字符串匹配度的平均值，然后使用匹配后的换位比例做衰减，无换位则不衰减，最多衰减0.5    
改进3 ：   (w1×m/s1+w2×m/s2) × (m-t)/m #字符串匹配度的**加权**平均值 ，权重与字符串的长度相关，字符串越长，权重越偏高，且w1+w2=1    

mode='popular' 流行方法  
mode='improve1' 改进方法1  
mode='improve3' 改进方法3  

![总说明](https://github.com/laura-zhang-cn/natural_language_preprocessing/blob/master/effect-images/jaro-winkler-%E4%BB%8B%E7%BB%8D%E5%9B%BE.png)
 
