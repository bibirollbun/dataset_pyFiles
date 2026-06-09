#first, let's undersand the prginal flow attention (without bias) from the paper:

'''
simple 1d seq implementation
https://github.com/thuml/Flowformer/blob/main/Flow_Attention.py
paper: Flowformer: Linearizing transformers with conservation flows. ICML, 2022a
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Flow_Attention(nn.Module):
    # flow attention in normal version
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=1e-6):
        super(Flow_Attention, self).__init__()
        pass
        self.n_heads = n_heads
        self.q_project = nn.Linear(d_input, d_model)
        self.k_project = nn.Linear(d_input, d_model)
        self.v_project = nn.Linear(d_input, d_model)
        self.out_project = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def dot_product(self, q, k, v):
        # q: B,h,L,dim   #dim = dim per head = d_model/h
        # k,v: B,h,S,dim

        kv = torch.einsum("nhld,nhlm->nhdm", k, v) #B,H,dim,dim
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)#B,h,L,dim
        return qkv

    def forward(self, queries, keys, values):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        # linear projection
        q = self.q_project(queries).view(B, L, self.n_heads, -1)
        k = self.q_project(keys).view(B, S, self.n_heads, -1)
        v = self.q_project(values).view(B, S, self.n_heads, -1)

        q = q.transpose(1, 2)#(B,h, L,  dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = torch.sigmoid(q)
        k = torch.sigmoid(k)

        ## Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1/(torch.einsum("nhld,nhd->nhl", q + self.eps, k.sum(dim=2) + self.eps))
        source_outgoing = 1/(torch.einsum("nhld,nhd->nhl", k + self.eps, q.sum(dim=2) + self.eps))

        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhd->nhl",
              q + self.eps,
              (k * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps
        )
        conserved_source = torch.einsum("nhld,nhd->nhl",
              k + self.eps,
              (q * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps
        )
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability

        # (3) Competition & Allocation
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])#B,h,S
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))#B,h,L

        x = self.dot_product(
            q * sink_incoming[:, :, :, None],  # for value normalization
            k,
            v * source_competition[:, :, :, None]
        ) #B,h,L,dim  #dim=dim per head = d_model/h

        x = (x* sink_allocation[:, :, :, None]
        ).transpose(1, 2)  # allocation #B,L,h,dim?

        ## (5) Final projection
        x = x.reshape(B, L, -1) #B,L,d_model
        x = self.out_project(x)
        x = self.dropout(x)
        return x #B,L,dim_out


d_input=4
d_model=6 #divisible by n_heads
d_output=5
n_heads=2

m = Flow_Attention(d_input, d_model, d_output, n_heads, drop_out=0.05, eps=1e-6)

B=1
S=5
L=6

queries =torch.randn(B, L, d_input)
keys=torch.randn(B, S, d_input)
values=torch.randn(B, S, d_input)

out = m(queries, keys, values)
print(out.shape)


### original triangle attnetion ------
def SignedSqrt( x):
    x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
    return x

# starting : outgoing edges : row attention
# see alphafold2 Supplementary Paper:
# https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf
# alogrithm 13 Algorithm 13 Triangular gated self-attention around starting node
# this is also implemented in drfold2
class TriAttStart(nn.Module):
    def __init__(self,z_dim, N_head=4, c=8):
        super(TriAttStart,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=nn.Linear(z_dim,c*N_head)
        self.klinear=nn.Linear(z_dim,c*N_head)
        self.vlinear=nn.Linear(z_dim,c*N_head)
        self.blinear=nn.Linear(z_dim,N_head)
        self.glinear=nn.Linear(z_dim,c*N_head)
        self.olinear=nn.Linear(c*N_head,z_dim)

    def forward(self,z):
        L,L,dim=z.shape  #L,L,D
        z = self.norm(z)
        q = self.qlinear(z).reshape(L,L,self.N_head,self.c)
        k = self.klinear(z).reshape(L,L,self.N_head,self.c)
        v = self.vlinear(z).reshape(L,L,self.N_head,self.c)
        b = self.blinear(z)#LLh
        att = torch.einsum('ijhc,ikhc->ijkh',q,k)*self.sq_c + b[None,:,:,:]
        att = F.softmax(SignedSqrt(att),dim=2)
        # if attdrop: #todo:droppout
        #     if self.training:
        #         att = basic.DropAtt(att,dim=2)
        o = torch.einsum('ijkh,ikhc->ijhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(L,L,self.N_head,self.c)) * o).reshape(L,L,-1)
        o = self.olinear(o)
        return o


#test
print('TriAttStart---')
L = 100
z_dim=32
z = torch.randn((L,L,z_dim))
attnetion = TriAttStart(z_dim)
out = attnetion(z)
print(out.shape)



##modified traingle attention from drfold2 with complexity O(L^2)

# figure.5 of the paper
class TriFlowAttStart(nn.Module):
    # flow attention in normal version
    def __init__(self,z_dim, N_head=4, c=8):
        super(TriFlowAttStart, self).__init__()
        self.eps = 1e-5
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c

        self.qlinear=nn.Linear(z_dim,c*N_head)
        self.klinear=nn.Linear(z_dim,c*N_head)
        self.vlinear=nn.Linear(z_dim,c*N_head)
        self.blinear=nn.Linear(z_dim,N_head)

    def forward(self,z):
        L, L, dim = z.shape
        # todo z = self.norm(z) group norm or layernorm ???
        # todo rope? do we embed rotationary pos??
        q = self.qlinear(z).reshape(L,L,self.N_head,self.c)
        k = self.klinear(z).reshape(L,L,self.N_head,self.c)
        v = self.vlinear(z).reshape(L,L,self.N_head,self.c)
        b = self.blinear(z)

        #we ignore efficiency and follow the paper order of dim
        q = q.permute(0,2,1,3) #L,h,L,c
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        b = b.permute(0, 2, 1,)#L,h,L,

        q1 = torch.cat([
            torch.sigmoid(q),
            torch.ones((L,self.N_head,L,1), device =q.device, dtype=q.dtype),
        ],-1) #L,L,h,c+1
        k1 = torch.cat([
            torch.sigmoid(k),
            b[...,None],
        ],-1)

        q1 = torch.sigmoid(q1)
        k1 = torch.sigmoid(k1)

        inv_I = 1/(torch.einsum('lhid,lhjd->lhi', q1, k1) +self.eps) # sum over j (3rd dim of Q, 1st dim of K)
        # Compute O: (L, H, L)
        inv_O =  1/(torch.einsum('lhjd,lhid->lhj', k1, q1) +self.eps)  # sum over i

        inv_I = inv_I[...,None] #L,h,L,1
        inv_O = inv_O[...,None]

        #eq 21 of paper
        #conserved_incoming_flow
        I_hat = torch.einsum('lhid,lhjd->lhi', q1, k1*inv_O) #L,h,L
        #conserved_outgoing_flow
        O_hat = torch.einsum('lhjd,lhid->lhj', k1, q1*inv_I) #L,h,L

        #from chatgpt ... not verified yet
        #eq 22 of paper
        # competition  # (L, H, L)
        v_hat = torch.softmax(O_hat, dim=-1).unsqueeze(-1) * v  # (L, h, L, c)
        
        # aggregation
        ###???? how to handle bias?
        kv = torch.einsum('lhjd,lhjd->lhd', k1[...,:-1], v_hat)  # (L, h, c)
        qkv = torch.einsum('lhid,lhd->lhid', q1[...,:-1], kv)  # (L, h, L, c)
        
        # Denominator: I = φ(Q) @ sum_j φ(K_j)^T
        I = torch.einsum('lhid,lhjd->lhi', q1[...,:-1], k1[...,:-1]) + self.eps
        att = qkv / I.unsqueeze(-1)  # (L, h, L, c)
        
        # allocation
        output = torch.sigmoid(I_hat).unsqueeze(-1)   * att  # (L, h, L, c)
        output = output.permute(0,2,1,3).reshape(L,L,-1)
        return output

print('TriFlowAttStart---')
L = 100
z_dim=32
z = torch.randn((L,L,z_dim))

attnetion = TriFlowAttStart(z_dim)
out = attnetion(z)
print(out.shape)
 


