import torch
import torch.nn as nn
import math



class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model    #维度512列数
        self.vocab_size=vocab_size      #行数——token词元数量
        self.embedding=nn.Embedding(vocab_size,d_model)  #构建词嵌入向量表   初始是随机的，后来通过训练进行优化，在222-224
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #生成位置编码所使用的公式
        pe=torch.zeros(seq_len,d_model)
        #创建公式分母部分
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)#（350，1）
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) #（1，256）
        pe[:,0::2]=torch.sin(position*div_term) #（350，256）
        pe[:,1::2]=torch.cos(position*div_term) #（Seq_len,d_model)
        pe=pe.unsqueeze(0)                      #(1,Seq_len,d_model)

        self.register_buffer('pe',pe)       #（这个注册为buffer，代表参数是确定的，不要再动了，以后训练也不会更新）

    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
#层归一化
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1)) #nn.Parameter 是专门为「可学习」而生的
        self.bias=nn.Parameter(torch.zeros(1)) #nn.Parameter 是专门为「可学习」而生的
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
#前馈网络
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)#W1 B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)#WE B2
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        # batch,seq_len,d_model-->batch,seq_len,d_ff-->batch,seq_len,d_model
        x=self.linear_1(x)
        x=torch.relu(x)
        x=self.dropout(x)
        x=self.linear_2(x)
        return x
class MuliHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model % h==0,'d_model is not divisable by h'
        self.d_k=d_model//h
        #接下来用于定义变换q,k,v的三个矩阵以及输出矩阵W_O
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        #'@'代表矩阵乘法     对最后两维度做转置
        #(batch,h,seq_len,d_k)-->(batch,h,seq_len,seq_len)
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        #  mask
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)

        attention_scores=attention_scores.softmax(dim=-1)#(batch,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value),attention_scores

    def forward(self,q,k,v,mask):
        query=self.w_q(q)#(Batch,Seq_len,d_model)-->(Batch,Seq_len,d_model)
        key=self.w_k(k)#(Batch,Seq_len,d_model)-->(Batch,Seq_len,d_model)
        value=self.w_v(v)#(Batch,Seq_len,d_model)-->(Batch,Seq_len,d_model)

        #torch.Tensor.view 的作用是：在不改变张量元素总数和数据顺序的前提下，把张量重新解释成另一个形状。它返回一个新的张量视图，通常与原张量共享底层内存。
        #(batch,seq_len,d_model)-->(batch,seq_len,h,d_k)-->(batch,h,seq_len.d_k)
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores=MuliHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        #已经得到每个注意力头的输出, 将每个头拼接起来
        #每个小头是（batch,h,seq_len,d_k) -->(batch,seq_len,h,d_k)-->(batch,seq_len,d_model)

        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        #(batch,seq_len,d_model)-->(batch,seq_len,d_model)
        return self.w_o(x)
class ResidualConnection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MuliHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self,x,src_mask):
        #第一条残差链接
        x=self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connection[1](x,self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MuliHeadAttentionBlock,cross_attention_block:MuliHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        #定义残差链接
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        #解码器自注意力用tgt_mask(含因果掩码,不能看未来); 交叉注意力用src_mask(只屏蔽padding)
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        #(batch,seq_len,d_model)-->(batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x),dim=-1)

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbedding,tgt_embed:InputEmbedding,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projecction_layer:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projecction_layer
    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    def project(self,x):
        return self.projection_layer(x)
def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,
                      d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048)->Transformer:
    #创建嵌入层
    src_embed=InputEmbedding(d_model,src_vocab_size)
    tgt_embed=InputEmbedding(d_model,tgt_vocab_size)
    #创建位置编码层
    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)
    #创建encoder_block
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MuliHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    #创建decoder_block
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MuliHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MuliHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
    #创建encoder decoder
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    #创建projectionLayers
    projection_layer=ProjectionLayer(d_model,tgt_vocab_size)
    #创建transformer
    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)
    #初始化参数
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer
