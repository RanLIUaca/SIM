import pandas as pd
import numpy as np
import multiprocessing as mp
from copy import deepcopy
import datetime
import time
import os


sleep_time = 0.1

# check the boundary and return the result
# if sel_index is out of the boundary, return 0
def get_value(sel_array, sel_index):
    sel_index = tuple(sel_index)
    inside = all([x>=0 and y>=0 and x<=(y-1) for x,y in zip(sel_index, sel_array.shape)])
    if inside:
        return(sel_array[sel_index])
    else:
        return(0.)


def read_para(store_path):
    init_dis = np.loadtxt(store_path+'_'+'init_dis.txt')
    trans_mat = np.loadtxt(store_path+'_'+'trans_mat.txt')
    px = np.loadtxt(store_path+'_'+'px.txt')
    py = np.loadtxt(store_path+'_'+'py.txt')
    pm = np.loadtxt(store_path+'_'+'pm.txt')
    pm = pm/pm.sum()
    p_mx = np.loadtxt(store_path+'_'+'p_mx.txt')
    p_my = np.loadtxt(store_path+'_'+'p_my.txt')
    dire_dis = []
    for i in range(trans_mat.shape[0]):
        dire_dis.append(np.loadtxt(store_path+'_'+str(i)+'dire_dis.txt'))

    return init_dis, trans_mat,px, py, pm, p_mx, p_my, dire_dis


# assign the states to each residue on the sequences
def assign_one_state_seq(seq_zip):
    state_seq, dire_seq = seq_zip[0], seq_zip[1]
    len_state = len(state_seq)
    x_state, y_state = [], []
    for i in range(len_state):
        for _ in range(dire_seq[i][0]):
            x_state.append(state_seq[i])
        for _ in range(dire_seq[i][1]):
            y_state.append(state_seq[i])
    return x_state, y_state


# compare the difference between two lists and calculate the number of the same elements
def comp_list(list1, list2, specific_state = None):
    if len(list1) == len(list2):
        len_list = len(list1)
        count_same = 0
        total_specific = 0
        for i in range(len_list):
            if list1[i] == list2[i]:
                if specific_state != None:
                    if list1[i] == specific_state:
                        count_same += 1
                        total_specific += 1
                else:
                    count_same += 1
        total_n = len_list
        return count_same, total_n
    else:
        raise ValueError('The length of list1 is not the same as that of list2.')


# calculate the difference between our estimated hidden parameters and the real ones
def diff_real_state_dire(est_state_seq, est_dire_seq, real_state_seq, real_dire_seq):
    real_res_state = list(map(assign_one_state_seq, zip(real_state_seq, real_dire_seq)))
    est_res_state = list(map(assign_one_state_seq, zip(est_state_seq, est_dire_seq)))
    n = len(real_res_state)
    count_same = 0
    total_n = 0
    for i in range(n):
        count_same_x, total_n_x = comp_list(real_res_state[i][0], est_res_state[i][0])
        count_same_y, total_n_y = comp_list(real_res_state[i][1], est_res_state[i][1])
        count_same = count_same + count_same_x + count_same_y
        total_n = total_n + total_n_x + total_n_y
    return count_same/total_n


# data1 is a pair of sequence data (letter), e.g., ['AACG','TTCG']
# full_data is a list of data similar to data1, e.g., [['AACG','TTCG'],['GCG','TCG']]
# N is the number of states
# init_dis: the initial distribution for states
# trans_mat: transition matrix
# emit_related_para : [p_x, p_y, p_m, p_mx, p_my], where in the symmetrical case p_my = p_mx
# mec_state: the mechanism of each state (only three possible values, M_j = 'x', 'y' or 'm')
# state_d: state direction, a list whose entry is a list of the possible direction
# e.g, [[(1,0)], [(0,1)], [(1,1),(1,2),(2,1)]]
# state_max_dx: the maximum index of directions for each state on x sequence
# state_max_dy: the maximum index of directions for each state on y sequence
# dire_dis: a list of the direction distribution for each state, where each entry is a two-dimensional array
# e.g., dire_dis[0] = np.array([[0,0],[1,0]]) represents, the direction is (1,0) with prob 1.
# obs2int_dict: a dictionary for observations, e.g., {'A':0,'T':1,'G':2,'C':3}
# state2int_dict: a dictionary for states, e.g., {'X':0 , 'Y':1, 'M':2}, for generating data
# all variables related to the states must be in the same order
# the first sequence 'x', the second sequence 'y'
obs2int_dict = dict()
aa_kinds = 'ARNDCQEGHILKMFPSTWYV'
for v, k in enumerate(aa_kinds):
    obs2int_dict[k] = int(v)

class sim_model:
    def __init__(self,  mec_state = ['x','y','m'], state_d= [[(1,0)],[(0,1)],[(1,1)]], 
        obs2int_dict = obs2int_dict, state2int_dict = {'X':0 , 'Y':1, 'M':2}, 
    init_dis = None, trans_mat = None, 
    px = None, py = None, pm = None, p_mx = None, p_my = None, 
    dire_dis = None, seed = None,
    trans_restrict = [(1,0)], sym_restrict = None, real_input = False):

        pesdu_count = 1e-6

        self.mec_state = mec_state
        self.state_d = state_d
        self.state_d_max = max(max(max(self.state_d)))
        self.obs2int_dict = obs2int_dict
        self.int2obs_dict = {v: k for k, v in self.obs2int_dict.items()}
        self.state2int_dict = state2int_dict
        self.int2state_dict = {v: k for k, v in self.state2int_dict.items()}
        
        if init_dis is None:
            if seed is None:
                self.init_dis = np.ones(len(mec_state))/len(mec_state)
            else:
                np.random.seed(seed)
                self.init_dis = np.random.sample(len(mec_state))
                self.init_dis = self.init_dis/self.init_dis.sum()
        else:
            init_dis = init_dis + pesdu_count
            self.init_dis = deepcopy(init_dis/init_dis.sum())
        
        if px is None:
            if seed is None:
                self.px = np.ones(len(obs2int_dict))/len(obs2int_dict)
            else:
                np.random.seed(seed+1)
                temp = np.random.sample(len(obs2int_dict))
                temp = temp/temp.sum()
                self.px = temp
        else:
            px = px + pesdu_count
            self.px = deepcopy(px/px.sum())

        if py is None:
            if seed is None:
                self.py = np.ones(len(obs2int_dict))/len(obs2int_dict)
            else:
                np.random.seed(seed+2)
                temp = np.random.sample(len(obs2int_dict))
                temp = temp/temp.sum()
                self.py = temp
        else:
            py = py + pesdu_count
            self.py = deepcopy(py/py.sum())
        
        if p_mx is None:
            if seed is None:
                self.p_mx = np.ones(len(obs2int_dict))/len(obs2int_dict)
            else:
                np.random.seed(seed+3)
                temp = np.random.sample(len(obs2int_dict))
                temp = temp/temp.sum()
                self.p_mx = temp
        else:
            p_mx = p_mx + pesdu_count
            self.p_mx = deepcopy(p_mx/p_mx.sum())

        if p_my is None:
            if seed is None:
                self.p_my = np.ones(len(obs2int_dict))/len(obs2int_dict)
            else:
                np.random.seed(seed+4)
                temp = np.random.sample(len(obs2int_dict))
                temp = temp/temp.sum()
                self.p_my = temp
        else:
            p_my = p_my + pesdu_count
            self.p_my = deepcopy(p_my/p_my.sum())

        if pm is None:
            if seed is None:
                self.pm = np.ones([len(obs2int_dict),len(obs2int_dict)])/(len(obs2int_dict)**2)
            else:
                np.random.seed(seed+5)
                temp = np.random.sample([len(obs2int_dict),len(obs2int_dict)])
                temp = temp/temp.sum()
                self.pm = temp
        else:
            pm = pm + pesdu_count
            self.pm = deepcopy(pm/pm.sum())
        
        # 'x-y,y-x' and 'm_x,m_y'
        self.trans_restrict = trans_restrict
        self.sym_restrict = sym_restrict
        if trans_mat is None:
            if seed is None:
                temp = np.ones([len(mec_state),len(mec_state)])
            else:
                np.random.seed(seed+6)
                temp = np.random.sample([len(mec_state),len(mec_state)])
            if trans_restrict is not None:
                for i,j in trans_restrict:
                    temp[i,j] = 0
            if sym_restrict is not None:
                for temp_pair in trans_restrict:
                    pos1, pos2 = temp_pair
                    temp = (temp.T/np.sum(temp,1)).T
                    ttpp = (temp[pos1[0],pos1[1]] + temp[pos2[0],pos2[1]])/2
                    temp[pos1[0],pos1[1]]  = ttpp
                    temp[pos2[0],pos2[1]] = ttpp
            temp = (temp.T/np.sum(temp,1)).T
            self.trans_mat = temp
        else:
            self.trans_mat = deepcopy(trans_mat)
        
        if dire_dis is None:
            self.dire_dis = [np.zeros([self.state_d_max+1,self.state_d_max+1]) for _ in range(len(state2int_dict))]
            for i in range(len(mec_state)):
                for lx, ly in state_d[i]:
                    self.dire_dis[i][lx,ly] = 1.
                self.dire_dis[i] = self.dire_dis[i]/np.sum(self.dire_dis[i])
        else:
            self.dire_dis = deepcopy(dire_dis)


        # check parameter
        if pm is not None:
            if np.abs(self.pm.sum() - 1.) > 0.00001:
                raise ValueError('The sum of pm should be 1!')


        if real_input:
            self.real_input = real_input
            self.input_real_para(init_dis, trans_mat, 
            px, py, pm, p_mx, p_my, dire_dis)

    def store_para(self, store_path):
        store_digit = '%.'+str(14)+'f'
        np.savetxt(store_path+'_'+'init_dis.txt', self.init_dis, fmt=store_digit)
        np.savetxt(store_path+'_'+'trans_mat.txt', self.trans_mat, fmt=store_digit)
        np.savetxt(store_path+'_'+'px.txt', self.px, fmt=store_digit)
        np.savetxt(store_path+'_'+'py.txt', self.py, fmt=store_digit)
        np.savetxt(store_path+'_'+'pm.txt', self.pm, fmt=store_digit)
        np.savetxt(store_path+'_'+'p_mx.txt', self.p_mx, fmt=store_digit)
        np.savetxt(store_path+'_'+'p_my.txt', self.p_mx, fmt=store_digit)
        for i in range(len(self.mec_state)):
            np.savetxt(store_path+'_'+str(i)+'dire_dis.txt', self.dire_dis[i], fmt=store_digit)

    # transform the string data to int data
    def trans_seq(self, data1):
        trans_data = []
        for i in range(len(data1)):
            temp_data = [self.obs2int_dict[j] for j in data1[i]]
            trans_data.append(temp_data)
        return trans_data
    
    def trans_all_seq(self, full_data):
        trans_full_data = list(map(self.trans_seq, full_data))
        return trans_full_data

    # select related entries for the kth state on the location sel_location 
    # the direction is sel_d: tuple
    # sel_location > 0, so it is different from the same index of trans_data
    # e.g., sel_location = (1,2), the first obs on X, the second obs on Y,
    # the corresponding index for trans_data is (0,1)
    def sel_entry(self, trans_data, sel_location, sel_d, obs_int = True):
        sel_obs = []
        sel_location = np.array(sel_location, dtype = int)
        # i=0: 'x'; i=1: 'y'
        
        for i in range(len(sel_d)):
            temp_sel = []
            for j in range(sel_d[i])[::-1]:
                temp_sel_index = int(sel_location[i] - j)
                # temp_sel_index = np.array(sel_location - j, dtype = int)[i]
                if(temp_sel_index<=0):
                    raise ValueError('wrong select elements!')
                if obs_int:
                    temp_sel.append(trans_data[i][temp_sel_index-1])
                else:
                    temp_sel.append(self.int2obs_dict[trans_data[i][temp_sel_index-1]])
            sel_obs.append(temp_sel)
        return sel_obs


    # calculate the probability of a direction under a specific state
    def cal_prob_dire(self, trans_data, sel_location, sel_d, state:int):
        result = 0
        temp_sel = self.sel_entry(trans_data, sel_location, sel_d)
        check_boundary = all(np.array(sel_location)>=np.array(sel_d))
        if check_boundary:
            if self.mec_state[state] == 'x':
                for i in range(sel_d[0]):
                    result += np.log(self.px[temp_sel[0][i]])
            elif self.mec_state[state] == 'y':
                for i in range(sel_d[1]):
                    result += np.log(self.py[temp_sel[1][i]])
            elif self.mec_state[state] == 'm':
                if sel_d[0] == 1:
                    result += np.log(self.p_mx[temp_sel[0][0]])
                    for i in range(sel_d[1]):
                        result += np.log(self.pm[temp_sel[0][0], temp_sel[1][i]
                        ]) - np.log(self.p_mx[temp_sel[0][0]])
                elif sel_d[1] == 1:
                    result += np.log(self.p_my[temp_sel[1][0]])
                    for i in range(sel_d[0]):
                        result += np.log(self.pm[temp_sel[0][i], temp_sel[1][0]
                        ]) - np.log(self.p_my[temp_sel[1][0]])
            return np.exp(result)
        else:
            return 0 
    
    def cal_one_alpha(self, trans_data):
        row_n, col_n = len(trans_data[0]), len(trans_data[1])
        state_n = len(self.mec_state)
        # add initial state first
        alpha = np.zeros([state_n+1, self.state_d_max+1, self.state_d_max+1, row_n+1, col_n+1])
        alpha[0,0,0,0,0] = 1
        for i in range(row_n+1):
            for j in range(col_n+1):
                if i==0 and j==0:
                    continue
                else:
                    for k in range(1,state_n+1):
                        for lx, ly in self.state_d[k-1]:
                            if i>=lx and j>=ly:
                                temp_obs_prob = self.cal_prob_dire(trans_data, [i,j], [lx,ly], state = k-1)
                                alpha[k,lx,ly,i,j] =  alpha[0,0,0,i-lx,j-ly]*self.init_dis[k-1
                                ]*temp_obs_prob*self.dire_dis[k-1][lx][ly]
                                for kk in range(1,state_n+1):
                                    for lx_star, ly_star in self.state_d[kk-1]:
                                        if i-lx>=lx_star and j-ly>=ly_star:
                                            alpha[k,lx,ly,i,j] += alpha[kk, lx_star, ly_star, i-lx, j-ly]*self.trans_mat[kk-1,
                                            k-1]*temp_obs_prob*self.dire_dis[k-1][lx][ly]
        return alpha
    

    def cal_one_viterbi(self, trans_data):
        row_n, col_n = len(trans_data[0]), len(trans_data[1])
        state_n = len(self.mec_state)
        # add initial state first
        w = np.zeros([state_n+1, self.state_d_max+1, self.state_d_max+1, row_n+1, col_n+1])
        w[0,0,0,0,0] = 1
        for i in range(row_n+1):
            for j in range(col_n+1):
                if i==0 and j==0:
                    continue
                else:
                    for k in range(1,state_n+1):
                        for lx, ly in self.state_d[k-1]:
                            if i>=lx and j>=ly:
                                # w[k,lx,ly,i,j]
                                temp_obs_prob = self.cal_prob_dire(trans_data, [i,j], [lx,ly], state = k-1)
                                temp_v = np.zeros([state_n+1, self.state_d_max+1, self.state_d_max+1])

                                # only i=lx,j=ly, will be non-zero
                                temp_v[0,0,0] =  w[0,0,0,i-lx,j-ly]*self.init_dis[k-1
                                ]*temp_obs_prob*self.dire_dis[k-1][lx][ly]
                                for kk in range(1,state_n+1):
                                    for lx_star, ly_star in self.state_d[kk-1]:
                                        if i-lx>=lx_star and j-ly>=ly_star:
                                            temp_v[kk, lx_star, ly_star] = w[kk, lx_star, ly_star, i-lx, j-ly]*self.trans_mat[kk-1,
                                            k-1]*temp_obs_prob*self.dire_dis[k-1][lx][ly]
                                
                                w[k,lx,ly,i,j] = np.max(temp_v)
        cur_pos = np.array([row_n, col_n])
        temp_w = w[:,:,:,cur_pos[0], cur_pos[1]]
        cur_state, cur_lx, cur_ly = np.unravel_index(np.argmax(temp_w, axis=None), temp_w.shape) 
        state_seq = [self.int2state_dict[cur_state-1]]
        int_state_seq = [cur_state-1]
        dire_seq = [(cur_lx, cur_ly)]
        while True:
            cur_pos = cur_pos - np.array([cur_lx, cur_ly])
            temp_w = w[:,:,:,cur_pos[0], cur_pos[1]]
            cur_state, cur_lx, cur_ly = np.unravel_index(np.argmax(temp_w, axis=None), temp_w.shape) 
            if cur_state == 0:
                break
            else:
                int_state_seq.insert(0, cur_state-1)
                state_seq.insert(0, self.int2state_dict[cur_state-1])
                dire_seq.insert(0, (cur_lx, cur_ly))
        
        if not all(cur_pos==np.array([0,0])):
            raise ValueError('final_pos is not (0,0)!')

        return int_state_seq, state_seq, dire_seq


    def cal_all_viterbi(self, trans_full_data, parallel = True, cores = 30):
        if parallel:
            pool = mp.Pool(processes = cores)
            all_viterbi = pool.map(self.cal_one_viterbi, trans_full_data)
            pool.close()
            pool.join(); time.sleep(sleep_time)
        else:
            all_viterbi = list(map(self.cal_one_viterbi, trans_full_data))
        
        all_int_state_seq, all_state_seq, all_dire_seq = [], [], []
        for i in all_viterbi:
            all_int_state_seq.append(i[0])
            all_state_seq.append(i[1])
            all_dire_seq.append(i[2])
        return all_int_state_seq, all_state_seq, all_dire_seq


    def cal_one_beta(self, trans_data):
        row_n, col_n = len(trans_data[0]), len(trans_data[1])
        state_n = len(self.mec_state)
        # add initial state first
        beta = np.zeros([state_n+1, self.state_d_max+1, self.state_d_max+1, row_n+1, col_n+1])
        for k in range(1,state_n+1):
            for lx, ly in self.state_d[k-1]:
                beta[k,lx,ly,row_n,col_n] = 1
        
        for i in range(row_n+1)[::-1]:
            for j in range(col_n+1)[::-1]:
                if i==row_n and j==col_n:
                    continue
                else:
                    for k in range(1,state_n+1):
                        for lx, ly in self.state_d[k-1]:
                            beta[k,lx,ly,i,j] =  0
                            for kk in range(1,state_n+1):
                                for lx_star, ly_star in self.state_d[kk-1]:
                                    if row_n>=i+lx_star and col_n>=j+ly_star:
                                        temp_obs_prob = self.cal_prob_dire(trans_data, [i+lx_star,j+ly_star], [
                                        lx_star,ly_star], state = kk-1)

                                        beta[k,lx,ly,i,j] += beta[kk, lx_star, ly_star, i+lx_star, j+ly_star
                                        ]*self.trans_mat[k-1,kk-1]*temp_obs_prob*self.dire_dis[kk-1][lx_star][ly_star]
        return beta


    def cal_one_nu(self, one_alpha, one_beta, cal_obs_lik = True):
        temp_nu = one_alpha*one_beta
        obs_lik = temp_nu[:,:,:,-1,-1].sum()
        nu = temp_nu/obs_lik
        if cal_obs_lik:
            return nu, obs_lik
        else:
            return nu

    def cal_one_lik(self, trans_data):
        temp_alpha = self.cal_one_alpha(trans_data)
        temp_beta = self.cal_one_beta(trans_data)
        _, temp_obs_lik = self.cal_one_nu(temp_alpha, temp_beta)
        return temp_obs_lik


    def cal_all_lik(self, trans_full_data, parallel = True, cores = 30):
        if parallel:
            pool = mp.Pool(processes = cores)
            all_lik = pool.map(self.cal_one_lik, trans_full_data)
            pool.close()
            pool.join(); time.sleep(sleep_time)
        else:
            all_lik = list(map(self.cal_one_lik, trans_full_data))
        return all_lik

    def cal_one_xi(self, trans_data, one_alpha, one_beta, obs_lik):
        row_n, col_n = len(trans_data[0]), len(trans_data[1])
        state_n = len(self.mec_state)
        temp_xi = np.zeros([state_n+1, state_n+1, row_n+1, col_n+1])
        for i in range(row_n+1):
            for j in range(col_n+1):
                for k in range(state_n+1):
                    for kk in range(1, state_n+1):
                        temp_xi[k,kk,i,j] = 0
                        for lx_star, ly_star in self.state_d[kk-1]:
                            if i>=lx_star and j>=ly_star: 
                                temp_obs_prob = self.cal_prob_dire(trans_data, [i,j], [
                                        lx_star,ly_star], state = kk-1)
                                if k==0:
                                    temp_comp = temp_obs_prob*one_beta[kk,lx_star,ly_star,i,j]*self.init_dis[kk-1
                                    ]*self.dire_dis[kk-1][lx_star][ly_star]
                                    temp_xi[k,kk,i,j] += one_alpha[0,0,0,i-lx_star,j-ly_star]*temp_comp
                                else:
                                    temp_comp = temp_obs_prob*one_beta[kk,lx_star,ly_star,i,j]*self.trans_mat[k-1,kk-1
                                    ]*self.dire_dis[kk-1][lx_star][ly_star]
                                    for lx, ly in self.state_d[k-1]:
                                        temp_xi[k,kk,i,j] += one_alpha[k,lx,ly,i-lx_star,j-ly_star]*temp_comp
        xi = temp_xi/obs_lik
        return xi
    

    def cal_one_nu_xi(self, trans_data, cal_obs_lik = True):
        one_alpha = self.cal_one_alpha(trans_data)
        one_beta = self.cal_one_beta(trans_data)
        one_nu, obs_lik = self.cal_one_nu(one_alpha, one_beta)
        one_xi = self.cal_one_xi(trans_data,one_alpha, one_beta, obs_lik)
        if cal_obs_lik:
            return one_nu, one_xi, obs_lik
        else:
            return one_nu, one_xi

    def cal_all_nu_xi(self, trans_full_data, parallel = True, cores = 30):
        if parallel:
            pool = mp.Pool(processes = cores)
            all_nu_xi = pool.map(self.cal_one_nu_xi, trans_full_data)
            pool.close()
            pool.join(); time.sleep(sleep_time)
        else:
            all_nu_xi = list(map(self.cal_one_nu_xi, trans_full_data))
        
        all_nu, all_xi, all_obs_lik = [], [], []
        for i in all_nu_xi:
            all_nu.append(i[0])
            all_xi.append(i[1])
            all_obs_lik.append(i[2])
        return all_nu, all_xi, all_obs_lik

    def count_obs(self, obs_list):
        obs_n = len(self.obs2int_dict)
        letter_count = np.zeros(obs_n,dtype=int)
        for i in obs_list:
            letter_count[i] += 1
        return letter_count

    # calculate hx, hy, hmx, hm
    def cal_one_h(self, trans_data):
        row_n, col_n = len(trans_data[0]), len(trans_data[1])
        state_n = len(self.mec_state)
        obs_n = len(self.obs2int_dict)
        hx, hy, hmx = [np.zeros([state_n+1, self.state_d_max+1, self.state_d_max+1, row_n+1, col_n+1, obs_n]) for _ in range(3)]
        hm = np.zeros([state_n+1, self.state_d_max+1, self.state_d_max+1, row_n+1, col_n+1, obs_n, obs_n])
        for i in range(row_n+1):
            for j in range(col_n+1):
                if i==0 and j==0:
                    continue
                else:
                    for k in range(1, state_n+1):
                        for lx, ly in self.state_d[k-1]:
                            # row_n>=i+lx and col_n>=j+ly and
                            if i>=lx and j>=ly:
                                temp_sel = self.sel_entry(trans_data, [i,j], [lx, ly])
                                mx = self.count_obs(temp_sel[0])
                                my = self.count_obs(temp_sel[1])
                                if self.mec_state[k-1] == 'x':
                                    hx[k, lx, ly, i, j,:] = mx
                                elif self.mec_state[k-1] == 'y':
                                    hy[k, lx, ly, i, j,:] = my
                                elif self.mec_state[k-1] == 'm':
                                    hmx[k, lx, ly, i, j,:] = (lx==1)*mx + (ly==1)*my - (lx==1 and ly==1)*mx*my
                                    for k1 in range(obs_n):
                                        for k2 in range(obs_n):
                                            if k1<=k2:
                                                hm[k, lx, ly, i, j, k1, k2] = (lx==1)*mx[k2]*my[k1] + (ly==1)*my[k2]*mx[k1
                                                ]-(lx==1 and ly==1 and k1==k2)*mx[k1]*my[k2]
        return hx, hy, hmx, hm

    def cal_all_h(self, trans_full_data, parallel = True, cores = 30):
        if parallel:
            pool = mp.Pool(processes = cores)
            all_h = pool.map(self.cal_one_h, trans_full_data)
            pool.close()
            pool.join(); time.sleep(sleep_time)
        else:
            all_h = list(map(self.cal_one_h, trans_full_data))
        all_hx, all_hy, all_hmx, all_hm = [], [], [], []
        for i in all_h:
            all_hx.append(i[0])
            all_hy.append(i[1])
            all_hmx.append(i[2])
            all_hm.append(i[3])
        return all_hx, all_hy, all_hmx, all_hm


    def pred_one_seq_state(self, trans_data):
        _, state_seq, dire_seq = self.cal_one_viterbi(trans_data)
        x_state, y_state = assign_one_state_seq((state_seq, dire_seq))
        final_x_state = ''.join(x_state)
        final_y_state = ''.join(y_state)
        final_state_seq = ''.join(state_seq)
        return final_x_state, final_y_state, final_state_seq

    def pred_all_seq_state(self, trans_full_data, parallel = True, cores = 30):
        if parallel:
            pool = mp.Pool(processes = cores)
            all_seq_state = pool.map(self.pred_one_seq_state, trans_full_data)
            pool.close()
            pool.join(); time.sleep(sleep_time)
        else:
            all_seq_state = list(map(self.pred_one_seq_state, trans_full_data))
        col_x, col_y, col_all = [], [], []
        for i in range(len(all_seq_state)):
            col_x.append(all_seq_state[i][0])
            col_y.append(all_seq_state[i][1])
            col_all.append(all_seq_state[i][2])
            # if i==1: 
            #     print(col_x);print(col_y);print(col_all)
        total_result = pd.DataFrame({'x_state': col_x, 'y_state': col_y, 'all_state':col_all})
        return total_result

    def update_para(self, trans_full_data,
    parallel = True, cores = 30, pesudo_count = 0.0001):
        state_n = len(self.mec_state)
        obs_n = len(self.obs2int_dict)

        # ss_time = time.time()
        all_nu, all_xi, all_obs_lik = self.cal_all_nu_xi(trans_full_data, parallel = parallel, cores = cores)
        all_hx, all_hy, all_hmx, all_hm = self.cal_all_h(trans_full_data, parallel = parallel, cores = cores)
        # ee_time = time.time()
        # print('parallel_time=',ee_time-ss_time)


        # update the initial distribution
        # ss_time = time.time()
        for j in range(state_n):
            self.init_dis[j] = sum([np.sum(xi[0,j+1,:,:]) for xi in all_xi])

        self.init_dis = self.init_dis+pesudo_count
        self.init_dis = self.init_dis/self.init_dis.sum()
        # ee_time = time.time()
        # print('initial_time=',ee_time-ss_time)

        # update transition matrix
        # ss_time = time.time()
        temp_trans_mat = np.zeros([state_n, state_n])
        for i in range(state_n):
            for j in range(state_n):
                temp_trans_mat[i,j] = sum([np.sum(xi[i+1,j+1,:,:]) for xi in all_xi])
        
        temp_trans_mat = temp_trans_mat + pesudo_count

        if self.trans_restrict is not None:
            for i, j in self.trans_restrict:
                    temp_trans_mat[i,j] = 0.
        if self.sym_restrict is not None:
            for temp_pair in self.sym_restrict:
                new_probx = np.zeros(state_n)
                new_proby = np.zeros(state_n)
                pos1, pos2 = temp_pair
                if pos1[0]!=pos2[0]:
                    nx = sum(temp_trans_mat[pos1[0],])
                    ny = sum(temp_trans_mat[pos2[0],])
                    cxy = temp_trans_mat[pos1[0],pos1[1]]
                    cyx = temp_trans_mat[pos2[0],pos2[1]]
                    new_probx[pos1[1]] = (cxy+cyx)/(nx+ny)
                    new_proby[pos2[1]] = (cxy+cyx)/(nx+ny)
                    for k in range(state_n):
                        if k!=pos1[1]:
                            new_probx[k] = temp_trans_mat[pos1[0],k]*(nx-cxy+ny-cyx)/(nx-cxy)/(nx+ny)
                    for k in range(state_n):
                        if k!=pos2[1]:
                            new_proby[k] = temp_trans_mat[pos2[0],k]*(nx-cxy+ny-cyx)/(ny-cyx)/(nx+ny)
                    temp_trans_mat[pos1[0],] = new_probx
                    temp_trans_mat[pos2[0],] = new_proby
                else:
                    ttpp = temp_trans_mat[pos1[0],pos1[1]] + temp_trans_mat[pos2[0],pos2[1]]
                    for k in range(state_n):
                        if k == pos1[1] or k == pos2[1]:
                            temp_trans_mat[pos1[0],k] = ttpp
                        else:
                            temp_trans_mat[pos1[0],k] = 2*temp_trans_mat[pos1[0],k]

        temp_trans_mat = (temp_trans_mat.T/np.sum(temp_trans_mat,1)).T
        self.trans_mat = temp_trans_mat
        # ee_time = time.time()
        # print('trans_time=',ee_time-ss_time)


        # update the distribution of directions under each state
        # ss_time = time.time()
        for k in range(state_n):
            for lx, ly in self.state_d[k]:
                self.dire_dis[k][lx,ly] = sum([np.sum(nu[k+1,lx,ly,:,:]) for nu in all_nu])
            self.dire_dis[k] = self.dire_dis[k] + pesudo_count
            self.dire_dis[k] = self.dire_dis[k]/np.sum(self.dire_dis[k])
        # ee_time = time.time()
        # print('dire_dis_time=',ee_time-ss_time)

        # update emition distribution
        # ss_time = time.time()
        for k in range(obs_n):
            self.px[k] = sum([np.sum(nu[:,:,:,:,:]*hx[:,:,:,:,:,k]) for nu,hx in zip(all_nu,all_hx)])
            self.py[k] = sum([np.sum(nu[:,:,:,:,:]*hy[:,:,:,:,:,k]) for nu,hy in zip(all_nu,all_hy)])
        self.px += pesudo_count
        self.py += pesudo_count
        self.px = self.px/self.px.sum()
        self.py = self.py/self.py.sum()

        # update emition distribution (conditional distribution)
        # ss_time = time.time()
        temp_conditional_pm = self.pm.copy()
        temp_conditional_pm[:] = 0
        for i in range(obs_n):
            for j in range(obs_n):
                if i<=j:
                    ttpp = sum([np.sum(nu[:,:,:,:,:]*hm[:,:,:,:,:,i,j]) for nu,hm in zip(all_nu,all_hm)])
                    if i==j:
                        temp_conditional_pm[i,j] = 2*ttpp
                    else:
                        temp_conditional_pm[i,j] = ttpp
                        temp_conditional_pm[j,i] = ttpp
        temp_conditional_pm += pesudo_count
        # self.pm = (temp_conditional_pm.T*self.p_mx).T
        self.pm = temp_conditional_pm/temp_conditional_pm.sum()
        self.p_mx = (self.pm@np.ones(obs_n).reshape(-1,1)).squeeze()
        self.p_my = (np.ones(obs_n).reshape(1,-1)@self.pm).squeeze()

        # ee_time = time.time()
        # print('conditional_pm_time=',ee_time-ss_time)

        # actually, it is the likelihood under the t-th parameter, while it is already (t+1)-th parameter now 
        # lag 1
        self.all_obs_lik = all_obs_lik


    # input the real parameters
    def input_real_para(self, init_dis, trans_mat, 
    px, py, pm, p_mx, p_my, dire_dis):
        self.real_init_dis = deepcopy(init_dis)
        self.real_trans_mat = deepcopy(trans_mat)
        self.real_px = deepcopy(px)
        self.real_py = deepcopy(py)
        self.real_p_mx = deepcopy(p_mx)
        self.real_p_my = deepcopy(p_my)
        self.real_pm = deepcopy(pm)
        self.real_dire_dis = deepcopy(dire_dis)
        self.real_input = True


    # calculate the difference between our estimations and the real ones
    def diff_real_para(self):
        error_para = {'init_dis':[], 'trans_mat':[], 
        'px':[], 'py':[], 'p_mx':[], 'p_my':[],
        'pm':[], 'dire_dis':[]}
        error_para['init_dis'].append(np.max(np.abs(self.init_dis - self.real_init_dis)))
        error_para['trans_mat'].append(np.max(np.abs(self.trans_mat - self.real_trans_mat)))
        error_para['px'].append(np.max(np.abs(self.px - self.real_px)))
        error_para['py'].append(np.max(np.abs(self.py - self.real_py)))
        error_para['p_mx'].append(np.max(np.abs(self.p_mx - self.real_p_mx)))
        error_para['p_my'].append(np.max(np.abs(self.p_my - self.real_p_my)))
        error_para['pm'].append(np.max(np.abs(self.pm - self.real_pm)))
        error_para['dire_dis'].append(np.max(np.abs(np.array(self.dire_dis) - np.array(self.real_dire_dis))))
        return error_para



    def error_record(self, trans_full_data, iterations = 1000,
    parallel = True, cores = 30, pesudo_count = 0.0001,
    store_path = '', store_para = False,
    real_state_seq = None, real_dire_seq = None):
        # check if the real parameters are already input
        loglik_trace = [] 
        if self.real_input:
            _, all_state_seq, all_dire_seq = self.cal_all_viterbi(trans_full_data, 
                    parallel = parallel, cores = cores)
            pred_hidden_ratio = diff_real_state_dire(all_state_seq, 
                    all_dire_seq, real_state_seq, real_dire_seq)
            ttpp_record = self.diff_real_para()
            ttpp_record['hidden_acc'] = pred_hidden_ratio
            error_para = pd.DataFrame(ttpp_record,index=[0,])
            for i in range(iterations):
                
                self.update_para(trans_full_data, 
                parallel = parallel, cores = cores, pesudo_count = pesudo_count)
                temp_loglik = np.sum(np.log(self.all_obs_lik))
                loglik_trace.append(temp_loglik)
                if i>50 and abs(loglik_trace[-2] - loglik_trace[-1]) < 0.0001:
                    break
                print(str(i)+':'+str(temp_loglik))
                print('init_dis_error:',ttpp_record['init_dis'])
                print('hidden_acc:',ttpp_record['hidden_acc'])
                if i != (iterations-1):
                    _, all_state_seq, all_dire_seq = self.cal_all_viterbi(trans_full_data, 
                    parallel = parallel, cores = cores)
                    pred_hidden_ratio = diff_real_state_dire(all_state_seq, 
                            all_dire_seq, real_state_seq, real_dire_seq)
                    ttpp_record = self.diff_real_para()
                    ttpp_record['hidden_acc'] = pred_hidden_ratio
                    temp_record = pd.DataFrame(ttpp_record,index=[i,])
                    error_para = pd.concat([temp_record, error_para], ignore_index = True) 
                error_para.to_csv(path_or_buf = store_path+
                '_'+'error_records.csv', index=False, doublequote = False)
                if store_para:
                    self.store_para(store_path)
                
            error_para['loglik'] = loglik_trace
            error_para.to_csv(path_or_buf = store_path+
            '_'+'error_records.csv', index=False, doublequote = False)
            
        else:
            raise ValueError('No real parameter input!')


    # the first list is x sequence, the second is y sequence
    def generate_direction_obs(self, state):
        temp_obs = [[],[]]
        temp_len = len(self.state_d[state])
        temp_prob = np.array([self.dire_dis[state][i[0]][i[1]] for i in self.state_d[state]])
        temp_dire = np.random.choice(temp_len, size = 1, p = temp_prob)[0]
        lx, ly = self.state_d[state][temp_dire]
        if self.mec_state[state] == 'x':
            for i in range(lx):
                emitted_obs = np.random.choice(len(self.px), size = 1, p = self.px)[0]
                temp_obs[0].append(self.int2obs_dict[emitted_obs]) 
        elif self.mec_state[state] == 'y':
            for i in range(ly):
                emitted_obs = np.random.choice(len(self.py), size = 1, p = self.py)[0]
                temp_obs[1].append(self.int2obs_dict[emitted_obs]) 
        elif self.mec_state[state] == 'm':
            if lx==1:
                emitted_obs = np.random.choice(len(self.p_mx), size = 1, p = self.p_mx)[0]
                temp_obs[0].append(self.int2obs_dict[emitted_obs]) 
                temp_prob = self.pm[emitted_obs,:]
                temp_prob = temp_prob/temp_prob.sum()
                for j in range(ly):
                    emitted_obs = np.random.choice(len(temp_prob), size = 1, p = temp_prob)[0]
                    temp_obs[1].append(self.int2obs_dict[emitted_obs]) 
            elif ly==1:
                emitted_obs = np.random.choice(len(self.p_my), size = 1, p = self.p_my)[0]
                temp_obs[1].append(self.int2obs_dict[emitted_obs]) 
                temp_prob = self.pm[:,emitted_obs]
                temp_prob = temp_prob/temp_prob.sum()
                for j in range(lx):
                    emitted_obs = np.random.choice(len(temp_prob), size = 1, p = temp_prob)[0]
                    temp_obs[0].append(self.int2obs_dict[emitted_obs])
        return (lx, ly), temp_obs

    # return a pair of sequences, state sequence, direction sequence
    def generate_one_data(self, state_len):
        state_n = len(self.mec_state)
        initial_state = np.random.choice(state_n, size = 1, p = self.init_dis)[0]
        cur_state = initial_state
        state_seq = [self.int2state_dict[initial_state]]
        int_state_seq = [initial_state]
        for i in range(state_len-1):
            temp_state = np.random.choice(state_n, size = 1, p = self.trans_mat[cur_state,:])[0]
            cur_state = temp_state
            state_seq.append(self.int2state_dict[temp_state])
            int_state_seq.append(temp_state)
        
        raw_result = list(map(self.generate_direction_obs, int_state_seq))
        obs_seq, dire_seq = [[],[]],[]
        for i in raw_result:
            dire_seq.append(i[0])
            obs_seq[0] += i[1][0]
            obs_seq[1] += i[1][1]
        x_obs_seq = ''.join(obs_seq[0])
        y_obs_seq = ''.join(obs_seq[1])

        return [x_obs_seq, y_obs_seq], state_seq, dire_seq

    def generate_data(self, state_len, data_size, seed=2022, parallel = True, cores = 30):
        np.random.seed(seed)
        state_len_list = [state_len]*data_size
        if parallel:
            pool = mp.Pool(processes = cores)
            all_data = pool.map(self.generate_one_data, state_len_list)
            pool.close()
            pool.join(); time.sleep(sleep_time)
        else:
            all_data = list(map(self.generate_one_data, state_len_list))
        seq_data, state_data, dire_data = [], [], []
        for i in all_data:
            seq_data.append(i[0])
            state_data.append(i[1])
            dire_data.append(i[2])

        return seq_data, state_data, dire_data

    def generate_one_neg_data(self, seq_len):
        x_seq = ''.join(np.random.choice(list(obs2int_dict.keys()), p = self.px,
                                               size=seq_len, replace = True))
        y_seq = ''.join(np.random.choice(list(obs2int_dict.keys()), p = self.py,
                                               size=seq_len, replace = True))
        return [x_seq, y_seq]

    def generate_neg_data(self, seq_len, data_size, seed=2022):
        np.random.seed(seed)
        neg_data = [self.generate_one_neg_data(seq_len) for _ in range(data_size)]
        return neg_data

    
    # the binding likelihood ratio 
    # binding ratio
    def cal_one_lr(self, trans_data):
        x_len, y_len = len(trans_data[0]), len(trans_data[1])
        emit_prob = 0

        for i in range(x_len):
            emit_prob += np.log(self.px[trans_data[0][i]])
        for i in range(y_len):
            emit_prob += np.log(self.py[trans_data[1][i]])
        emit_prob = np.exp(emit_prob)
        temp_lik = self.cal_one_lik(trans_data)
        result = temp_lik/(emit_prob+temp_lik)
        return result
    
    def cal_all_lr(self, trans_full_data, parallel = True, cores = 30):
        if parallel:
            pool = mp.Pool(processes = cores)
            all_lr = pool.map(self.cal_one_lr, trans_full_data)
            pool.close()
            pool.join(); time.sleep(sleep_time)
        else:
            all_lr = list(map(self.cal_one_lr, trans_full_data))
        return all_lr
    
# raw_data is a dataframe: the first column is X sequence, the second column is Y sequence. 
# store_all_seeds: True is to return all estimation results of seeds; 
# False is to only return the result of the seed which maxmize the observed likelihood.
def est_para(raw_data, store_path = '',
mec_state = ['x','y','m'], state_d= [[(1,0)],[(0,1)],[(1,1)]], 
obs2int_dict = obs2int_dict, state2int_dict = {'X':0 , 'Y':1, 'M':2}, 
trans_restrict=[(1,0)],
seeds = None, store_all_seeds = True, parallel = True, cores = 30, iterations = 1000):
    seq_data = []
    store_digit = '%.'+str(10)+'f'
    lik_result = 0
    for _, i in raw_data.iterrows():
        seq_data.append([i[0],i[1]])
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    print('Training begins with the seed:', 'unif')
    model1 = sim_model(mec_state,  state_d, obs2int_dict, state2int_dict,
        trans_restrict = trans_restrict)
    trans_all_data = model1.trans_all_seq(seq_data)
    
    loglik_trace = [] 
    for i in range(iterations):
        model1.update_para(trans_all_data, parallel = parallel, cores = cores)
        if i%10==0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
            print('Current seed:'+'unif'+'; current iteration:'+str(i))
        temp_loglik = np.sum(np.log(model1.all_obs_lik))
        loglik_trace.append(temp_loglik)
        model1.store_para(os.path.join(store_path,'unif'))
        np.savetxt(os.path.join(store_path,'unif')+'_'+'lik.txt', loglik_trace, fmt=store_digit)
        if i>50 and abs(loglik_trace[-2] - loglik_trace[-1]) < 1e-6:
            break

    model1.store_para(os.path.join(store_path,'unif'))
    np.savetxt(os.path.join(store_path,'unif')+'_'+'lik.txt', loglik_trace, fmt=store_digit)

    state_result = model1.pred_all_seq_state(trans_all_data, parallel = parallel, cores = cores)
    state_result.to_csv(os.path.join(store_path,'unif')+'_'+'train_states.csv', index=False, doublequote = False)

    if not store_all_seeds:
        model1.store_para(os.path.join(store_path,'ml'))
        state_result.to_csv(os.path.join(store_path,'ml')+'_'+'train_states.csv', index=False, doublequote = False)
        lik_result = loglik_trace[-1]
        sel_seed = 'unif'

    if seeds is not None:
        for seed in seeds:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
            print('Training begins with the seed:', str(seed))
            model1 = sim_model(mec_state,  state_d, obs2int_dict, state2int_dict,
                trans_restrict = trans_restrict,seed = seed)
            loglik_trace = [] 
            for i in range(iterations):
                model1.update_para(trans_all_data, parallel = parallel, cores = cores)
                if i%10==0:
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
                    print('Current seed:'+str(seed)+'; current iteration:'+str(i))
                temp_loglik = np.sum(np.log(model1.all_obs_lik))
                loglik_trace.append(temp_loglik)
                model1.store_para(os.path.join(store_path,str(seed)))
                np.savetxt(os.path.join(store_path,str(seed))+'_'+'lik.txt', loglik_trace, fmt=store_digit)
                if i>50 and abs(loglik_trace[-2] - loglik_trace[-1]) < 1e-6:
                    break

            model1.store_para(os.path.join(store_path,str(seed)))
            np.savetxt(os.path.join(store_path,str(seed))+'_'+'lik.txt', loglik_trace, fmt=store_digit)
            state_result = model1.pred_all_seq_state(trans_all_data, parallel = parallel, cores = cores)
            state_result.to_csv(os.path.join(store_path,str(seed))+'_'+'train_states.csv', index=False, doublequote = False)

            if not store_all_seeds:
                temp_lik = loglik_trace[-1]
                if temp_lik>lik_result:
                    model1.store_para(os.path.join(store_path,'ml'))
                    state_result.to_csv(os.path.join(store_path,'ml')+'_'+'train_states.csv', index=False, doublequote = False)
                    lik_result = temp_lik
                    sel_seed = str(seed)

    if not store_all_seeds:
        print('Max_lik_seed:', str(sel_seed))
        with open(os.path.join(store_path,'ml')+'_'+'sel_seed.txt', 'w') as f:
            f.write(sel_seed)



# train_data and test_data are both dataframes: the first column is X sequence, 
# the second column is Y sequence. 
# all_seed_pred: True is to use all seeds to predict; False is to only use the seed of max likelihood
def pred_test_lr_score(train_data, test_data, all_seed_pred = False, store_path = '',
mec_state = ['x','y','m'], state_d= [[(1,0)],[(0,1)],[(1,1)]], 
obs2int_dict = obs2int_dict, state2int_dict = {'X':0 , 'Y':1, 'M':2}, 
trans_restrict=[(1,0)], seeds = None, parallel = True, cores = 30, 
iterations = 1000, use_existing_est = False, pred_store_path = None):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    if not use_existing_est:
        print('Training begins.')
        est_para(train_data, store_path = store_path,
        mec_state = mec_state, state_d = state_d, 
        obs2int_dict = obs2int_dict, state2int_dict = state2int_dict, 
        trans_restrict=trans_restrict,
        seeds = seeds, store_all_seeds = all_seed_pred, 
        parallel = parallel, cores = cores, iterations = iterations)
    else:
        print('Use previous estimations.')

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    print('Predicting begins.')
    if not all_seed_pred:
        print('The seed of max likelihood is predicting.')
        store_seed_path = os.path.join(store_path,'ml')
        init_dis, trans_mat, px, py, pm, p_mx, p_my, dire_dis = read_para(store_seed_path)
        model1 = sim_model(mec_state = mec_state, state_d= state_d, 
            obs2int_dict = obs2int_dict, state2int_dict = state2int_dict, 
            init_dis = init_dis, trans_mat = trans_mat, 
            px = px, py = py, pm = pm, p_mx = p_mx, p_my = p_my, 
            dire_dis = dire_dis, trans_restrict = trans_restrict)
        seq_data = []
        for _, i in test_data.iterrows():
            seq_data.append([i[0],i[1]])
        trans_all_test_data = model1.trans_all_seq(seq_data)
        pred_score = model1.cal_all_lr(trans_all_test_data, parallel = parallel, cores = cores)
        state_result = model1.pred_all_seq_state(trans_all_test_data, parallel = parallel, cores = cores)
        temp_result = test_data.copy()
        temp_result['pred_score'] = pred_score
        if pred_store_path is None:
            state_result.to_csv(store_seed_path+'_'+'pred_states.csv', index=False, doublequote = False)
            temp_result.to_csv(store_seed_path+'_'+'pred_score.csv', index=False, doublequote = False)
        else:
            pred_seed_path = os.path.join(pred_store_path,'ml')
            state_result.to_csv(pred_seed_path+'_'+'pred_states.csv', index=False, doublequote = False)
            temp_result.to_csv(pred_seed_path+'_'+'pred_score.csv', index=False, doublequote = False)
    else:
        if seeds is None:
            seeds = np.array([])
        seeds = np.append(seeds, 'unif')
        for seed in seeds:
            print('Seed',str(seed),'predicting.')
            store_seed_path = os.path.join(store_path,str(seed))

            init_dis, trans_mat, px, py, pm, p_mx, p_my, dire_dis= read_para(store_seed_path)
            model1 = sim_model(mec_state = mec_state, state_d= state_d, 
                obs2int_dict = obs2int_dict, state2int_dict = state2int_dict, 
                init_dis = init_dis, trans_mat = trans_mat, 
                px = px, py = py, pm = pm, p_mx = p_mx, p_my = p_my,
                dire_dis = dire_dis, trans_restrict = trans_restrict)

            seq_data = []
            for _, i in test_data.iterrows():
                seq_data.append([i[0],i[1]])
            trans_all_test_data = model1.trans_all_seq(seq_data)
            pred_score = model1.cal_all_lr(trans_all_test_data, parallel = parallel, cores = cores)
            state_result = model1.pred_all_seq_state(trans_all_test_data, parallel = parallel, cores = cores)
            temp_result = test_data.copy()
            temp_result['pred_score'] = pred_score
            if pred_store_path is None:
                state_result.to_csv(store_seed_path+'_'+'pred_states.csv', index=False, doublequote = False)
                temp_result.to_csv(store_seed_path+'_'+'pred_score.csv', index=False, doublequote = False)
            else:
                pred_seed_path = os.path.join(pred_store_path,str(seed))
                state_result.to_csv(pred_seed_path+'_'+'pred_states.csv', index=False, doublequote = False)
                temp_result.to_csv(pred_seed_path+'_'+'pred_score.csv', index=False, doublequote = False)
    print('Succeed.')
