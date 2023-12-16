![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/5980e23c-9ac8-4275-98cc-2647db4b64dc)# img2LaTeX
## 构建模型
参考了GitHub用户kingyiusuen的代码仓库image-to-latex
###使用的模型的backbone是ResNet+Transformer。其结构图如下：
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/70fa91a9-6d44-4e62-883a-4b2244441047)
（图源：Singh, Sumeet S. and Sergey Karayev. “Full Page Handwriting Recognition via Image to Sequence Extraction.” ArXiv abs/2103.06450 (2021): n. pag.）

### Encoder部分：
加载 ResNet-18 模型，并将其不同层组织成一个顺序神经网络，作为编码器的主干网络。使用一个卷积层 bottleneck 将输入特征图的通道数从 256 减少到 d_model，以适应后续的处理。应用了一个二维位置编码器 image_positional_encoder，用于对输入的图像特征进行位置编码。其结构如下：
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/717cba6c-9cef-412b-a327-c4e49d090cd6)
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/ead52ca0-c5a4-46c8-8ed5-67707451bc40)

### Decoder部分：
使用一个嵌入层 embedding 将类别标签映射为对应的嵌入向量，用于表示文本输入。创建了一个遮罩张量 y_mask，用于在解码过程中限制模型只依赖已生成的输出。创建了一个一维位置编码器 word_positional_encoder，用于对解码器的输入进行位置编码。使用 Transformer 解码器层 TransformerDecoderLayer 创建了一个多层的 Transformer 解码器 transformer_decoder，用于生成输出序列。最后，通过一个线性层 fc 将解码器的输出映射到类别标签的数量上，进行分类。其结构如下图：
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/d6a94c84-4882-4485-8f95-c44ccd5f8b74)
（图源：Singh, Sumeet S. and Sergey Karayev. “Full Page Handwriting Recognition via Image to Sequence Extraction.” ArXiv abs/2103.06450 (2021): n. pag.）

###模型优化
优化方面主要探究了ResNet层数对于预测结果的影响，以及探索了Beam-search和长句惩罚对于预测结果的优化。
Pytorch的官方代码中提供了五种不同深度结构的ResNet神经网络，分别为18、34、50、101、152层。
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/a6cd2056-0a4f-4398-925d-a9d2043f6323)
（图源：参考论文Deep Residual Learning for Image Recognition）

其中ResNet18（3个Cov Block）+Transformer进行18个epoch的训练表现如下：
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/5ad0c4f6-dfaa-4158-bdb7-4d4bb859c896)
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/a1db2d19-c0a5-4449-8acb-32a43f34d327)

ResNet34（3个Cov Block）+Transformer进行18个epoch的训练表现如下：
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/0a5da07a-b6c6-466a-ae66-61e9cf0eaaec)
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/361260bb-112c-42e6-8536-0abc8f735d3a)

ResNet50（4个Cov Block）+Transformer进行18个epoch的训练的表现如下：
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/8426667b-a0fa-4f91-9442-7140d9ca7a0d)
![图片](https://github.com/Justjustifyjudge/img2LaTeX/assets/117241737/a6da6c4a-e2ed-46a9-b934-e7cbb908500c)
其在验证集上的错字率下降到了0.020。
在目标任务的基础上，使用ResNet34相较于ResNet18能一定程度地提高模型的表现，但是提高得较为有限。
ResNet50的测试结果相较于ResNet34又有一点点提高，但是提高的效果不明显，甚至不如改变随机种子的提高大。认为结果可能存在偶然性，不能说明ResNet50效果更优。
同时ResNet50需要更大的显存和内存空间进行运算，在代码不进行优化的情况下会超出kaggle分配的内存空间导致优化失败。

同时我们对模型使用的predict函数进行了两方面的优化尝试：
原版本的参考代码对于预测时没有任何优化的，是直接使用decode函数然后挑选其中逻辑概率最高的结果返回。我们考虑到A.每次都搜索概率最高的情况，可能会掉入局部最优的陷阱，B.由于少量长句我们必须将超参数的max_len调大来防止输入时的张量溢出，而在进行预测时绝大部分的预测结果理想情况下都应该为中句或者短句，因此有必要引入长度惩罚的机制。我们尝试了以下两种方法，其中beam_search并没有取得优化的结果。
A.使用beam-search来一定程度上避免掉入局部最优解的情况。代码如下：
#resnet_transformer.py
######################################## Beam 的示例化和选前top个候选的函数
class Beam:
    def __init__(self,indices,score):
        self.indices=indices
        self.score=score
        self.has_ended=(indices[:-1]==self.eos_index).type_as(self.indices)

def generate_candidates(self,beams,top_tokens):
    new_beams=[]
    for beam in beams:
        if beam.has_ended:
            new_beams.append(beam)
        else:
            for token in top_tokens:
                new_indices=torch.cat((beam.indices,token.unsqueeze(0)),dim=-1)
                new_score=beam.score+scores[token.item()]
                new_beam=Beam(new_indices,new_score)
                new_beams.append(new_beam)
    return new_beams
#######################################

def predict(self, x: Tensor) -> Tensor:
        """Make predctions at inference time.

        Args:
            x: (B, C, H, W). Input images.

        Returns:
            (B, max_output_len) with elements in (0, num_classes - 1).
        """
        B = x.shape[0]
        S = self.max_output_len
        #########################################
        # 改用bean_search
        K=3
        encoded_x=self.encode(x)
        
        output_indices=torch.full((B,K,S),self.pad_index).type_as(x).long()
        output_indices[:,:,0]=self.sos_index
        has_ended=torch.full((B,K),False)
        
        for Sy in range(1,S):
            candidate_indices=output_indices[:,:,:Sy]
            logits=self.decode(candidate_indices.view(B*K,Sy),encoded_x)
            logits=logits.view(B,K,-1)
            scores=torch.log_softmax(logits,dim=-1)
            scores=scores[:,:,-1]
            
            # 找钱K个概率大的token
            top_scores, toptokens=torch.topk(scores,K,dim=-1)
            
            # 创建新的Beam_search束
            new_beams=[]
            for i in range(B):
                beams=[]
                for j in range(K):
                    beam=Beam(output_indices[i,j],top_score[i,j])
                    beams.append(beam)
                new_beams.extend(self.generate_candidates(beams,top_tokens[i]))
            
            # 为new beams排个序
            new_beams.sort(key=lambda b:b.score,reverse=True)
            
            output_indices=torch.full((B,K,S),self.pad_index).type_as(x).long()
            has_ended=torch.full((B,K),False)
            for j,beam in enumerate(new_beams[:K]):
                output_indices[:,j]=beam.indices
                has_ended[:j]=beam.has_ended
            
            if torch.all(has_ended):
                break
        
        # 最后把选出来的token组成钜子
            eos_positions=find_first(output_indices[:,0],self.eos_index)
            for i in range(B):
                j=int(eos_positions[i].item())+1
                output_indices[i,:,j:]=self.pad_index
                
            return output_indices[:,0,:]
        
        #########################################

B.使用GNMT的长度惩罚方式，规定length_penalty表示处罚的权重。部分代码如下：
for Sy in range(1, S):
            y = output_indices[:, :Sy]  # (B, Sy)
            logits = self.decode(y, encoded_x)  # (Sy, B, num_classes)
            # Select the token with the highest conditional probability
            output = torch.argmax(logits, dim=-1)  # (Sy, B)
            output_indices[:, Sy] = output[-1:]  # Set the last output token

            # Early stopping of prediction loop to speed up prediction
            has_ended |= (output_indices[:, Sy] == self.eos_index).type_as(has_ended)
            if torch.all(has_ended):
                break
        # Set all tokens after end token to be padding
        eos_positions = find_first(output_indices, self.eos_index)

        for i in range(B):
            j = int(eos_positions[i].item()) + 1
            output_indices[i, j:] = self.pad_index

        lengths = (output_indices != self.pad_index).sum(dim=-1).type_as(x)
        lp_lengths = ((5.0 + lengths) / 6.0) ** length_penalty
        lp_output_indices = output_indices.float() / lp_lengths.unsqueeze(-1)

        return lp_output_indices.long()

### 参数优化
调参时超参数的设置主要从epoch、seed、注意力头数nhead、decoder_layer、batch_size、图像transform处理、学习率lr几个方向出发进行修改。
epoch的多少，影响模型训练是否充分。
在epoch=3的时候，模型在验证集上只能取得58.265的总体性能分，到epoch=15的时候可以取得90.164的性能分，进一步的，到epoch=20的时候，可以取得91.805的性能分。
随机种子seed的设置，采取了42、1234、3407等选取的几个数字，不能判断该超参数的选取应该采取什么策略，不过似乎1234是许多种子中训练效果比较好的。
nhead的设置受限于embedding层的大小，尝试了4、8、16几个，其中4个和8个的表现比较好。
learning_rate设置为0.0005相较于0.0001有利于跳出局部最优，把验证集上得到的综合性能分再提升两分。
