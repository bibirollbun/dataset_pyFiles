#todo: pip install


import os,sys

from scipy.optimize import fmin_l_bfgs_b,fmin_cg,fmin_bfgs
from scipy.optimize import minimize
import potential_fold.lbfgs_rosetta as lbfgs_rosetta

# potention fold
import potential_fold.Potential as Potential
import potential_fold.Cubic as Cubic
import potential_fold.operations as operations
import potential_fold.rigid as rigid
import potential_fold.a2b as a2b

import torch
import torch.autograd as autograd
import numpy as np
import pickle
from timeit import default_timer as timer

import matplotlib 
import matplotlib.pyplot as plt


#helper function
def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)
	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)
	else:
		raise NotImplementedError

print('IMPORT OK !!!')


NUM_DIST_BIN=38

# reference location
POTENTIAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__))) +'/potential_fold'
BASE  = ["P", "C4'", "N1"]
OTHER = ["O5'", "C5'", "C3'", "O3'", "C1'"]
SIDE  = ["N1", "C2", "O2", "N2", "N3", "N4", "C4", "O4", "C5", "C6", "O6", "N6", "N7", "N8", "N9"]
BASE_COOR  = np.load(f'{POTENTIAL_DIR}/lib/base.npy') #shape=(4,3,3) : (A, G, C, U) x (P,C4,N) x (xyz)
OTHER_COOR = np.load(f'{POTENTIAL_DIR}/lib/other2.npy')#shape=(4,5,3) :
SIDE_COOR  = np.load(f'{POTENTIAL_DIR}/lib/side.npy')#shape=(4,15,3) :
# missing is xyz=(0,0,0)

VDW_weight={
    "weight_pp" : 0,
    "weight_cc" : 0,
    "weight_nn" : 0,
    "weight_pccp": 0,
    "weight_cnnc": 0,
    "weight_pnnp": 0,
    "weight_pcc" :0,
    "weight_cnn": 0,
    "weight_pnn": 0,
    "weight_vdw": 0.5,
    "weight_nn_contact":0,
    "weight_cc_contact":0,
    "weight_beta": 0,
    "weight_fape": 0,
    "weight_bond": 100
}
DEFAULT_WEIGHT={
    "weight_pp": 1,
    "weight_cc": 1,
    "weight_nn": 1,
    "weight_pccp": 0,
    "weight_cnnc": 0,
    "weight_pnnp": 0,
    "weight_pcc": 0,
    "weight_cnn": 0,
    "weight_pnn": 0,
    "weight_vdw": 1,
    "weight_nn_contact": 0,
    "weight_cc_contact": 0,
    "weight_beta": 0,
    "weight_fape": 2,
    "weight_bond": 5000,
    "pair_weight_power": 0.25,
    "pair_weight_min": 0.2,
    "pair_error_power": 3,
    "pair_rest_min_dist": 2,
    "FAPE_max": 30,
    "geo_scale": 450,
    "num_of_models": 5
}

print('CONFIG OK !!!')


# io helper
def read_data_optimizer(
    netout_dir,
    model_id
):
    #netout key: plddt(L,L), coor(L,3,3), dist_p(L,L,38), dist_c, dist_n,
    #pLDDT = "Predicted Local Distance Difference Test"
    netout = []
    for i in model_id:
        pickle_file = f'{netout_dir}/model_{i:02d}.netout.pickle'
        with open(pickle_file, 'rb') as f:
            out = pickle.load(f)
        netout.append(out)

    return netout


class Optimizer(object):
    def __init__(self, seq, target_id, netout, best_netout_index=0):
        pass
        self.scale_factor=1 #learning rate
        self.best_netout_index=best_netout_index

        self.target_id = target_id
        self.seq = seq
        self.L = len(self.seq)

        self.netout = netout

        self.pair = []
        self.txs = []
        for out in self.netout:
            self.pair.append( torch.from_numpy(out['plddt']).double()  )
            self.txs.append( torch.from_numpy( out['coor'] ).double() )
        self.num_out = len(self.netout)

        self.basex  = operations.Get_base(self.seq, BASE_COOR) #L,3,3
        self.otherx = operations.Get_base(self.seq, OTHER_COOR) #L,5,3
        self.sidex  = operations.Get_base(self.seq, SIDE_COOR) #L,15,3

        # aligned tx to basex
        self.tx2ds=[]
        for tx in self.txs:
            true_rot,true_trans = operations.Kabsch_rigid(self.basex,tx[:,0],tx[:,1],tx[:,2])
            true_x2 = tx[:,None,:,:] - true_trans[None,:,None,:]
            true_x2 = torch.einsum('ijnd,jde->ijne',true_x2,true_rot.transpose(-1,-2))
            self.tx2ds.append(true_x2)

        self.geos = []
        for out in self.netout:
            geo = {}
            geo['pp'] = out['dist_p'].astype(np.float64)
            geo['cc'] = out['dist_c'].astype(np.float64)
            geo['nn'] = out['dist_n'].astype(np.float64)
            self.geos.append(geo)

        #make cubic spline interpolator for discrete distance distribution
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        for i,geo in enumerate(self.geos):
            print('\r',f'Optimizer.__init__(): {i} fitting cubic distance ... ', end='', flush=True) #take sometime
            cc_cs, cc_decs = Cubic.dis_cubic(geo['cc'], 2, 40, NUM_DIST_BIN-2) # min, max, num_bin
            pp_cs, pp_decs = Cubic.dis_cubic(geo['pp'], 2, 40, NUM_DIST_BIN-2)
            nn_cs, nn_decs = Cubic.dis_cubic(geo['nn'], 2, 40, NUM_DIST_BIN-2)
            self.geo_cc.append([cc_cs, cc_decs])
            self.geo_pp.append([pp_cs, pp_decs])
            self.geo_nn.append([nn_cs, nn_decs])
        print('\n','done!')

        self.make_mask()

        self.local_weight = torch.ones(self.L,self.L) # fape of neighboring pairs control the torsion
        for i in range(self.L):
            for j in range(i+1,min(self.L,i+2)):
                self.local_weight[i,j] = self.local_weight[j,i] = 4
            for j in range(i+2,min(self.L,i+3)):
                self.local_weight[i,j] = self.local_weight[j,i] = 3
            for j in range(i+3,min(self.L,i+4)):
                self.local_weight[i,j] = self.local_weight[j,i] = 2
        zz=0

    #initialisation
    def make_mask(self):
        halfmask=np.zeros([self.L,self.L])
        fullmask=np.zeros([self.L,self.L])
        for i in range(self.L):
            for j in range(i+1,self.L):
                halfmask[i,j]=1
                fullmask[i,j]=1
                fullmask[j,i]=1

        self.halfmask=torch.DoubleTensor(halfmask) > 0.5
        self.fullmask=torch.DoubleTensor(fullmask) > 0.5
        self.clash_mask = torch.zeros([self.L,self.L,22,22])  ##todo: why 22?

        for i in range(self.L):
            for j in range(i+1,self.L):
                self.clash_mask[i,j]=1

        for i in range(self.L):
             self.clash_mask[i,i,:6,7:]=1

        for i in range(self.L-1):
            self.clash_mask[i,i+1,:,0]=0
            self.clash_mask[i,i+1,0,:]=0
            self.clash_mask[i,i+1,:,5]=0
            self.clash_mask[i,i+1,5,:]=0

        self.side_mask  = rigid.side_mask(self.seq)
        self.side_mask  = self.side_mask[:,None,:,None] * self.side_mask[None,:,None,:]
        self.clash_mask = (self.clash_mask > 0.5) * (self.side_mask > 0.5)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []

        for geo in self.geos:
            confimask_cc = torch.DoubleTensor(geo['cc'][:,:,-1]) < 0.5
            confimask_pp = torch.DoubleTensor(geo['pp'][:,:,-1]) < 0.5
            confimask_nn = torch.DoubleTensor(geo['nn'][:,:,-1]) < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)



    # quaternion-based rotation
    def init_quat(self, i):
        x = torch.rand([self.L, 21])
        x[:, 18:] = self.txs[i].mean(dim=1)
        init_coor = self.txs[i]
        biasq = torch.mean(init_coor, dim=1, keepdim=True)
        q = init_coor - biasq
        m = torch.einsum('bnz,bny->bzy', self.basex, q).reshape([self.L, -1])

        x[:, :9] = x[:, 9:18] = m
        x.requires_grad = True
        return x

    def init_quat_safe(self, i):
        x = torch.rand([self.L, 21])
        x[:, 18:] = self.txs[i].mean(dim=1)
        init_coor = self.txs[i]
        biasq = torch.mean(init_coor, dim=1, keepdim=True)
        q = init_coor - biasq + torch.rand([self.L, 3, 3])
        m = (torch.einsum('bnz,bny->bzy', self.basex, q) + torch.eye(3)[None, :, :]).reshape([self.L, -1])

        x[:, :9] = x[:, 9:18] = m
        x.requires_grad = True
        return x

    #------------
    # compute all clash
    def compute_bb_clash(self,coor,other_coor):
        com_coor = torch.cat([coor,other_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 3.15) * (self.clash_mask)
        #vdw_dynamic=torch.nn.functional.softplus(3.15-com_dis[dynamicmask2_vdw])

        # Lennard-Jones (LJ) potential for van der Waals interactions (attractive and repulsive forces between atoms).
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],3.15)
        return vdw_dynamic.sum()*self.config['weight_vdw']

    def compute_full_clash(self,coor,other_coor,side_coor):
        com_coor = torch.cat([coor[:,:2],other_coor,side_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 2.5) * (self.clash_mask)
        #vdw_dynamic=torch.nn.functional.softplus(3.15-com_dis[dynamicmask2_vdw])
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],2.5)
        return vdw_dynamic.sum()*self.config['weight_vdw']


    ## compute energy ----
    def compute_cc_energy(self, coor):
        min_dis,max_dis,bin_num = 2,40,NUM_DIST_BIN-2 #36
        c_atoms = coor[:,1]
        upper_th = max_dis -((max_dis-min_dis)/bin_num)*0.5
        lower_th=3.10
        cc_map = operations.pair_distance(c_atoms,c_atoms)

        total_ecb = 0 #C4'-C4' backbone interaction energy
        for cc_cs, confimask_cc in zip(self.geo_cc,self.geo_confimask_cc):
            dynamicmask_cc = (cc_map <= upper_th) * (confimask_cc) * (self.fullmask) * (cc_map >= 2.5)
            dynamicmask_cc_np = dynamicmask_cc.numpy()  #todo : fix change from torch to np array
            if dynamicmask_cc_np.sum()>1:
                cc_coe=torch.DoubleTensor(np.array([acs.c for acs in cc_cs[0][dynamicmask_cc_np]]))
                cc_x = torch.DoubleTensor(np.array([acs.x for acs in cc_cs[0][dynamicmask_cc_np]]))
                E_cb = Potential.cubic_distance(cc_map[dynamicmask_cc],cc_coe,cc_x,min_dis,max_dis,bin_num).sum()*self.config['weight_cc']*0.5
            else:
                E_cb =0
            E_cb = E_cb +  ((  (cc_map <= 2.5)* (self.fullmask) * (confimask_cc) *5  ).sum() )*self.config['weight_cc']
            total_ecb = total_ecb + E_cb
        return total_ecb

    def compute_pp_energy(self,coor):
        min_dis,max_dis,bin_num = 2,40,NUM_DIST_BIN-2
        p_atoms=coor[:,0]
        upper_th=max_dis - ((max_dis-min_dis)/bin_num)*0.5
        lower_th=3.10
        pp_map=operations.pair_distance(p_atoms,p_atoms)

        total_ecb = 0
        for pp_cs,confimask_pp in zip(self.geo_pp,self.geo_confimask_pp):
            dynamicmask_pp= (pp_map <= upper_th) * (confimask_pp) * (self.fullmask) * (pp_map >= 2.5)
            dynamicmask_pp_np=dynamicmask_pp.numpy()
            if dynamicmask_pp_np.sum()>1:
                pp_coe=torch.DoubleTensor(np.array([acs.c for acs in pp_cs[0][dynamicmask_pp_np]]))
                pp_x = torch.DoubleTensor(np.array([acs.x for acs in pp_cs[0][dynamicmask_pp_np]]))
                E_cb = Potential.cubic_distance(pp_map[dynamicmask_pp],pp_coe,pp_x,min_dis,max_dis,bin_num).sum()*self.config['weight_pp']*0.5
            else:
                E_cb =0
            E_cb =E_cb +  ((     (pp_map <= 2.5)* (self.fullmask) * (confimask_pp)   *5   ).sum() )*self.config['weight_pp']
            total_ecb = total_ecb + E_cb
        return total_ecb

    def compute_nn_energy(self,coor):
        min_dis,max_dis,bin_num = 2,40,NUM_DIST_BIN-2
        n_atoms=coor[:,-1]
        upper_th=max_dis - ((max_dis-min_dis)/bin_num)*0.5
        lower_th=3.10
        nn_map=operations.pair_distance(n_atoms,n_atoms)

        total_ecb = 0
        for nn_cs,confimask_nn in zip(self.geo_nn,self.geo_confimask_nn):
            dynamicmask_nn= (nn_map <= upper_th) * (confimask_nn) * (self.fullmask) * (nn_map >= 2.5)
            dynamicmask_nn_np=dynamicmask_nn.numpy()
            if dynamicmask_nn_np.sum()>1:
                nn_coe=torch.DoubleTensor(np.array([acs.c for acs in nn_cs[0][dynamicmask_nn_np]]))
                nn_x = torch.DoubleTensor(np.array([acs.x for acs in nn_cs[0][dynamicmask_nn_np]]))
                E_cb = Potential.cubic_distance(nn_map[dynamicmask_nn],nn_coe,nn_x,min_dis,max_dis,bin_num).sum()*self.config['weight_nn']*0.5
            else:
                E_cb =0
            E_cb = E_cb +  ((    (nn_map <= 2.5)* (self.fullmask) * (confimask_nn)  *5    ).sum() )*self.config['weight_nn']
            total_ecb = total_ecb + E_cb
        return total_ecb

    def compute_pccp_energy(self,coor):
        p_atoms=coor[:,0]
        c_atoms=coor[:,1]
        pccpmap = operations.dihedral( p_atoms[self.pccpi], c_atoms[self.pccpi], c_atoms[self.pccpj] ,p_atoms[self.pccpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pccp_coe,self.pccp_x,36)
        return neg_log.sum()*self.config['weight_pccp']

    def compute_cnnc_energy(self,coor):
        n_atoms=coor[:,-1]
        c_atoms=coor[:,1]
        pccpmap=operations.dihedral( c_atoms[self.cnnci], n_atoms[self.cnnci], n_atoms[self.cnncj] ,c_atoms[self.cnncj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.cnnc_coe,self.cnnc_x,36)
        return neg_log.sum()*self.config['weight_cnnc']

    def compute_pnnp_energy(self,coor):
        n_atoms=coor[:,-1]
        p_atoms=coor[:,0]
        pccpmap=operations.dihedral( p_atoms[self.pnnpi], n_atoms[self.pnnpi], n_atoms[self.pnnpj] ,p_atoms[self.pnnpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pnnp_coe,self.pnnp_x,36)
        return neg_log.sum()*self.config['weight_pnnp']

    def compute_pcc_energy(self,coor):
        p_atoms=coor[:,1]
        c_atoms=coor[:,2]
        pccmap=operations.angle( p_atoms[self.pcci], c_atoms[self.pcci], c_atoms[self.pccj]                   )
        neg_log = Potential.cubic_angle(pccmap,self.pcc_coe,self.pcc_x,12)
        return neg_log.sum()*self.config['weight_pcc']

    def compute_fape_energy(self,coor,ep=1e-3,epmax=20):
        energy= 0
        for tx in self.tx2ds:
            px_mean = coor[:,[1]]
            p_rot   = operations.rigidFrom3Points(coor)
            p_tran  = px_mean[:,0]
            pred_x2 = coor[:,None,:,:] - p_tran[None,:,None,:] # Lx Lrot N , 3
            pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
            errmap = torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum(  torch.clamp(errmap,max=epmax)        )
        return energy * self.config['weight_fape']

    def reweight_func(self,ww):
        reweighting = torch.pow(ww,self.config['pair_weight_power'])
        reweighting[ww < self.config['pair_weight_min']] = 0
        return reweighting

    def compute_fape_energy_fromquat(self,x,coor,ep=1e-6,epmax=100):
        energy= 0
        p_rot,px_mean = a2b.Non2rot(x[:,:9],x.shape[0]),x[:,9:]
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
        #coor  = a2b.quat2b(x)
        for tx,weightplddt in zip(self.tx2ds,self.pair):
            # px_mean = coor[:,[1]]
            # p_rot   = operations.rigidFrom3Points(coor)
            # p_tran  = px_mean[:,0]

            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            #energy = energy + torch.sum(  torch.clamp(errmap,max=epmax) * self.reweight_func(weightplddt[...,None])    )
            #energy = energy + torch.sum(  self.tooth_func(errmap) * weightplddt[...,None]    )
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) * self.local_weight[...,None] )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )
        return energy * self.config['weight_fape']

    # oxygen-phosphate : RNA phosphodiesterlinkage
    def compute_bond_energy(self,coor, other_coor):
        # 3.87
        o3 = other_coor[:-1,-2]
        p  = coor[1:,0]
        dis = (o3-p).norm(dim=-1)
        energy = ((dis-1.607)**2).sum()
        return energy * self.config['weight_bond']

    #final energy!
    def energy(self, rama):
        #todo: why some divided by E, some don't ????
        coor       = a2b.quat2b(self.basex, rama[:, 9:])
        other_coor = a2b.quat2b(self.otherx, rama[:, 9:])
        side_coor  = a2b.quat2b(self.sidex, torch.cat([rama[:, :9], coor[:, -1]], dim=-1))
        # print(coor.shape,other_coor.shape,side_coor.shape)

        E_cc, E_pp, E_nn = 0,0,0
        E_pccp, E_cnnc, E_pnnp = 0,0,0,
        E_vdw  = 0
        E_fape = 0
        E_bond = 0

        if self.config['weight_cc'] > 0:
            E_cc = self.compute_cc_energy(coor) / self.num_out

        if self.config['weight_pp'] > 0:
            E_pp = self.compute_pp_energy(coor) / self.num_out

        if self.config['weight_nn'] > 0:
            E_nn = self.compute_nn_energy(coor) / self.num_out

        if self.config['weight_pccp'] > 0:
            E_pccp = self.compute_pccp_energy(coor) / self.num_out

        if self.config['weight_cnnc'] > 0:
            E_cnnc = self.compute_cnnc_energy(coor) / self.num_out

        if self.config['weight_pnnp'] > 0:
            E_pnnp = self.compute_pnnp_energy(coor) / self.num_out

        if self.config['weight_vdw'] > 0:
            E_vdw = self.compute_full_clash(coor, other_coor, side_coor)

        if self.config['weight_fape'] > 0:
            E_fape = self.compute_fape_energy_fromquat(rama[:, 9:], coor) / self.num_out

        if self.config['weight_bond'] > 0:
            E_bond = self.compute_bond_energy(coor, other_coor)

        return E_vdw + E_fape + E_bond + E_pp + E_cc + E_nn + E_pccp + E_cnnc + E_pnnp

    ####################################################################3
     # optimization function
    def obj_func_grad_np(self,rama):
        rama = torch.DoubleTensor(rama)

        rama.requires_grad=True
        if rama.grad:
            rama.grad.zero_()
        f=self.energy(rama.view(self.L,21))*self.scale_factor
        grad_value=autograd.grad(f,rama)[0]
        return grad_value.data.numpy().astype(np.float64)

    def obj_func_np(self,rama):
        rama = torch.DoubleTensor(rama)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f=self.energy(rama)*self.scale_factor
            #print('score',f)
            return f.item()

    def do_optimize(self):
        minenergy = 1e16
        minirama = None
        if True:
            # for ilter in range( len(self.txs)):
            ilter = self.best_netout_index
            try:
                rama = self.init_quat(ilter).data.numpy()
            except:
                rama = self.init_quat_safe(ilter).data.numpy()

            for i in range(3):
                self.config = VDW_weight
                rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama, fprime=self.obj_func_grad_np, iprint=10)[0]
                rama = rama.flatten()

                self.config = DEFAULT_WEIGHT
                geoscale = self.config['geo_scale']
                self.config['weight_pp'] = geoscale * self.config['weight_pp']
                self.config['weight_cc'] = geoscale * self.config['weight_cc']
                self.config['weight_nn'] = geoscale * self.config['weight_nn']
                self.config['weight_pccp'] = geoscale * self.config['weight_pccp']
                self.config['weight_cnnc'] = geoscale * self.config['weight_cnnc']
                self.config['weight_pnnp'] = geoscale * self.config['weight_pnnp']
                for i in range(3):
                    rama = fmin_l_bfgs_b(
                        func=self.obj_func_np, x0=rama, fprime=self.obj_func_grad_np,
                        iprint=10
                    )[0]

                    line_min = lbfgs_rosetta.ArmijoLineMinimization(
                        self.obj_func_np, self.obj_func_grad_np,
                        True, len(rama), 120
                    )
                    lbfgs_opt = lbfgs_rosetta.lbfgs(self.obj_func_np, self.obj_func_grad_np)
                    rama = lbfgs_opt.run(
                        rama, 256, lbfgs_rosetta.absolute_converge_test,
                        line_min, 8000, self.obj_func_np,
                        self.obj_func_grad_np, 1e-9
                    )
                    rama = fmin_l_bfgs_b(
                        func=self.obj_func_np, x0=rama, fprime=self.obj_func_grad_np,
                        iprint=10
                    )[0]

                newrama = rama + 0.0
                newrama = torch.DoubleTensor(newrama)
                current_energy = self.obj_func_np(rama)
                # self.outpdb(newrama,self.saveprefix+f'_{str(ilter)}'+'.pdb',energystr=str(current_energy))

                if current_energy < minenergy:
                    print(current_energy, minenergy)
                    minenergy = current_energy
                    #<todo>!self.outpdb(newrama, self.saveprefix + '.pdb', energystr=str(current_energy))



#example here !!!!
if 1:
    netout_dir = '/media/hp/c30d34ed-0d55-4077-82dc-b56cd13dd548/2025/kaggle/stanford-rna-3d-folding/code/dummy01/study/dummy-data/netout'
    model_id = [9,8,7] #e.g. sorted by a scoring function


    netout = read_data_optimizer(
        netout_dir,
        model_id
    )
    target_id = 'R1116'
    seq = 'CGCCCGGAUAGCUCAGUCGGUAGAGCAGCGGCUAAAACAGCUCUGGGGUUGUACCCACCCCAGAGGCCCACGUGGCGGCUAGUACUCCGGUAUUGCGGUACCCUUGUACGCCUGUUUUAGCCGCGGGUCCAGGGUUCAAGUCCCUGUUCGGGCGCCA'

    start_timer = timer()
    folding_optimizer = Optimizer(seq, target_id, netout, best_netout_index=0)
    print('Optimizer:', time_to_str(timer()-start_timer, mode='min')) #20 models take 04 min
    #exit(0)

    start_timer = timer()
    folding_optimizer.do_optimize()
    print('do_optimize:', time_to_str(timer()-start_timer, mode='min'))


#visualise result

