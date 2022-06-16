#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:02:52 2021

@author: apple
"""

import pandas as pd
import openpyxl
from pandas import ExcelWriter
from datetime import datetime
import copy
import cvxpy as cv
import numpy as np


#%% Importing parameters

params_list = pd.read_excel("_master.xlsx", sheet_name="Parameters")

params = {}
for p in params_list.columns:
    params[p] = {}
    params[p]['Unit'] = params_list.loc[0,p]
    params[p]['Values'] = [value for value in params_list.loc[1:,p] if not pd.isna(value)]
 
support_path = 'Data_entry - Support.xlsx'
support = openpyxl.load_workbook(support_path)
data_sheets = support.sheetnames


#%% Start loop

metadata = {}
database = {}
counter = 0

MuInd = pd.MultiIndex.from_product([params['User']['Values'],
                                    params['Location']['Values'],
                                    params['Yearly mileage']['Values'],
                                    params['qi']['Values'],
                                    params['Electricity price']['Values'],
                                    params['BEV incentives']['Values'],
                                    params['BEV capacity']['Values'], 
                                    params['Gasoline Price']['Values']],
                                    names=['User','Location','Yearly mileage','Time Horizon',
                                           'Electricity price',' BEV Incentives','BEV capacity', 'Gasoline Price'])


Res=pd.DataFrame(0,index=MuInd,columns=["output"])


for user in params['User']['Values']:
    database[user] = {}
    house_surface = params['House surface']['Values'][params['User']['Values'].index(user)]
    
    for location in params['Location']['Values']:
        database[user][location] = {}

        for mileage in params['Yearly mileage']['Values']:
            database[user][location][mileage] = {}
           
            for year in params['qi']['Values']:
                database[user][location][mileage][year] = {}
    
                for ee_price in params['Electricity price']['Values']:
                    database[user][location][mileage][year][ee_price] = {}
                
                    for bev_incentive in params['BEV incentives']['Values']:
                        database[user][location][mileage][year][ee_price][bev_incentive] = {}

                        for gas_price in params['Gasoline Price']['Values']:
                            database[user][location][mileage][year][ee_price][bev_incentive][gas_price]= {} 
                            
                            for bev_capacity in params['BEV capacity']['Values']:
                            
                                bev_model = params['BEV model']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_price = params['BEV price']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_max_km = params['BEV max km per hour']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_cons = params['BEV consumption']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                icev_model = params['ICEV model']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                icev_max_km = params['ICEV max km per hour']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                icev_price = params['ICEV price']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                icev_emiss = params['ICEV emissions']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_v2h_capacity = params['BEV V2H capacity']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_v2h_model = params['BEV V2H model']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_v2h_price = params['BEV V2H price']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_v2h_max_km = params['BEV V2H max km per hour']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                bev_v2h_cons = params['BEV V2H consumption']['Values'][params['BEV capacity']['Values'].index(bev_capacity)]
                                
                                "Filling metadata"
                                metadata[counter] = {}
                                metadata[counter]['User'] = user
                                metadata[counter]['House surface'] = house_surface
                                metadata[counter]['Location'] = location
                                metadata[counter]['Yearly mileage'] = mileage
                                metadata[counter]['BEV capacity'] = bev_capacity
                                metadata[counter]['BEV model'] = bev_model
                                metadata[counter]['BEV price'] = bev_price
                                metadata[counter]['BEV max km per hour'] = bev_max_km
                                metadata[counter]['ICEV model'] = icev_model
                                metadata[counter]['ICEV price'] = icev_price
                                metadata[counter]['BEV V2H capacity'] = bev_v2h_capacity
                                metadata[counter]['BEV V2H model'] = bev_v2h_model
                                metadata[counter]['BEV V2H price'] = bev_v2h_price
                                metadata[counter]['BEV V2H max km per hour'] = bev_v2h_max_km
                                metadata[counter]['Gasoline Price'] = gas_price
                                
            
                                "Creating new data_entry, one per iteration"
                                data_entry = ExcelWriter(r'inputs\data_entry\Data_entry_{}.xlsx'.format(str(counter)))
                                
                                for sheet in data_sheets:
                            
                                    if sheet=='n-t':
                                        support_sheet = pd.read_excel(support_path, sheet_name = sheet, header=[0,1])
                                        # support_sheet[("Technology","Unit")][support_sheet[("Need","Name")]=="Electricity"][support_sheet[("Technology","Name")]=="BEV"] = bev_model                        
                                        support_sheet.iloc[3,3] = bev_model
                                        support_sheet.iloc[4,3] = bev_v2h_model
                                        support_sheet.iloc[10,3] = bev_model
                                        support_sheet.iloc[11,3] = bev_v2h_model
                            
                                    if sheet=='qi':
                                        support_sheet = pd.read_excel(support_path, sheet_name = sheet, header=None, index_col=[0])
                                        
                                        support_sheet.iloc[1,0] = datetime(support_sheet.iloc[0,0].year+year,
                                                                           support_sheet.iloc[0,0].month, 
                                                                           support_sheet.iloc[0,0].day+1,
                                                                           23, 00, 00)
                                    
                                    if sheet=='f':
                                        support_sheet = pd.read_excel(support_path, sheet_name = sheet, header=None, index_col=[0])
                                        support_sheet.columns = [support_sheet.iloc[0,0]]
                                        support_sheet.index.names = [list(support_sheet.index)[0]]
                                        support_sheet = support_sheet.iloc[1:,:]
                                        
                                    if sheet=='l':
                                        support_sheet = pd.read_excel(support_path, sheet_name = sheet, header=[0], index_col=[0])
                                        support_sheet.columns = [support_sheet.iloc[0,0]]
                                        support_sheet.index.names = [list(support_sheet.index)[0]]
                                        support_sheet = support_sheet.iloc[1:,:]
                        
                                    if sheet=='i':
                                        support_sheet = pd.read_excel(support_path, sheet_name = sheet, header=[0,1], index_col=[0])
                                    if sheet=='qo':
                                        support_sheet = pd.read_excel(support_path, sheet_name = sheet, header=None, index_col=[0])
                                    if sheet=='additional parameters':
                                        support_sheet = pd.read_excel(support_path, sheet_name = sheet, header=[0], index_col=[0])
                                       
                                        
                                    support_sheet.to_excel(data_entry, sheet_name=sheet) 
                                    
                                data_entry.save()
                                data_entry.close()
                                                    
                                
                                "Reading new data_entry"
                                
                                # Data Entry
                                input_nt = pd.read_excel(r'inputs\data_entry\Data_entry_{}.xlsx'.format(counter), header=[0,1], index_col=[0], sheet_name='n-t')
                                
                                input_f = pd.read_excel(r'inputs\data_entry\Data_entry_{}.xlsx'.format(counter), header=[0,1], sheet_name='f')
                                
                                input_l = pd.read_excel(r'inputs\data_entry\Data_entry_{}.xlsx'.format(counter), header=[0], sheet_name='l')
                                input_l.columns = pd.MultiIndex.from_arrays([["Satellite account", "Satellite account"],list(input_l.columns)])
                                
                                input_i = pd.read_excel(r'inputs\data_entry\Data_entry_{}.xlsx'.format(counter), header=[0,1], sheet_name='i').iloc[1:,:]
                                input_i.index = [i for i in range(input_i.shape[0])]
                                
                                input_time_operation = pd.read_excel(r'inputs\data_entry\Data_entry_{}.xlsx'.format(counter), header=[0], index_col=[0], sheet_name='qo')
                                input_time_investment = pd.read_excel(r'inputs\data_entry\Data_entry_{}.xlsx'.format(counter), header=[0], index_col=[0], sheet_name='qi')
                                
                                # Names of group of variables
                                Nn = input_nt.columns[0][0]
                                Nt = input_nt.columns[2][0]
                                Ny = '{} Type'.format(Nt)
                                Na = 'Activity'
                                Nc = 'Commodity'
                                Nf = input_f.columns[0][0]
                                Nl = input_l.columns[0][0]
                                Ni = input_i.columns[0][0]
                                Nqo = input_time_operation.iloc[2,0]
                                Nqi = input_time_investment.iloc[2,0]
                                
                                # Introducing activities (a) and commodities (c)
                                input_nt[(Na,'Name')] = 0
                                input_nt[(Na,'Unit')] = 0
                                input_nt[(Nc,'Name')] = 0
                                input_nt[(Nc,'Unit')] = 0
                                
                                # Naming and giving the unit
                                for ii in input_nt.index:
                                    input_nt.loc[ii,(Na,'Name')] = 'Exploiting {} for {}'.format(input_nt.loc[ii,(Nt,'Name')],input_nt.loc[ii,('Need','Name')])
                                    input_nt.loc[ii,(Na,'Unit')] = '{}'.format(input_nt.loc[ii,('Need','Unit')])
                                    input_nt.loc[ii,(Nc,'Name')] = '{} from {}'.format(input_nt.loc[ii,('Need','Name')],input_nt.loc[ii,(Nt,'Name')])
                                    input_nt.loc[ii,(Nc,'Unit')] = '{}'.format(input_nt.loc[ii,('Need','Unit')])
                                        
                                # Getting all the needed indeces
                                n = input_nt.loc[:,'Need'].drop_duplicates() # needs
                                t = input_nt.loc[:,Nt].drop_duplicates() # technologies
                                a = input_nt.loc[:,Na].drop_duplicates() # activities
                                c = input_nt.loc[:,Nc].drop_duplicates() # commodities
                                bat = t.loc[t.Type=='Storage'].reset_index(drop=True) #subset of storage technologies
                                sol = t.loc[t.Type=='Solar'].reset_index(drop=True) #subset of solar technologies
                                f = input_f.loc[:,Nf].drop_duplicates() # factor of production
                                l = input_l.loc[:,Nl].drop_duplicates() # satellite account
                                i = input_i.loc[:,Ni].drop_duplicates() # technology information
                                qo = pd.date_range(start=input_time_operation.loc['Data inizio'].values[0], end=input_time_operation.loc['Data fine'].values[0], freq=input_time_operation.loc['Frequenza'].values[0]) # time step for operational decisions
                                qi = pd.date_range(start=input_time_investment.loc['Data inizio'].values[0], end=input_time_investment.loc['Data fine'].values[0], freq=input_time_investment.loc['Frequenza'].values[0]) # time step for investment decisions
                                
                                # Adding the charging activities of the storage technologies
                                for ii in input_nt.index:
                                    if input_nt.loc[ii,(Nt,'Type')] == 'Storage':
                                        a = a.append({'Name': 'Charging {}'.format(input_nt.loc[ii,(Nt,'Name')]),
                                                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
                                        c = c.append({'Name': '{} Charge'.format(input_nt.loc[ii,(Nt,'Name')]),
                                                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
                                        l = l.append({'Name': 'Charge from {} '.format(input_nt.loc[ii,(Nt,'Name')]),
                                                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
                                    if input_nt.loc[ii,(Nt,'Type')] == 'Solar':
                                        a = a.append({'Name': 'Selling {} surplus'.format(input_nt.loc[ii,(Nt,'Name')]),
                                                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()
                                        c = c.append({'Name': 'Sold surplus from {}'.format(input_nt.loc[ii,(Nt,'Name')]),
                                                      'Unit':'{}'.format(input_nt.loc[ii,(Nt,'Type')])},ignore_index=True).drop_duplicates()                    
                                
                                
                                "Building the needed matrices that do not need any interaction with the user"
                                
                                # G, the matrix that connects needs and commodities
                                n_index = pd.MultiIndex.from_product([list(n.loc[:,'Name'])], names=['Need'])
                                c_index = pd.MultiIndex.from_product([list(c.loc[:,'Name'])], names=[Nc])
                                G = pd.DataFrame(0, index=n_index, columns=c_index)
                                for ni in n.loc[:,'Name']:
                                    for ci in c.loc[:,'Name']:
                                         if ci.find(ni)!=-1: G.loc[ni,ci] = 1
                                
                                # J, the matrix that technologies and activity
                                t_index = pd.MultiIndex.from_product([list(t.loc[:,'Name'])], names=[Nt])
                                bat_index = pd.MultiIndex.from_product([list(bat.loc[:,'Name'])], names=['Storage'])
                                sol_index = pd.MultiIndex.from_product([list(sol.loc[:,'Name'])], names=['Solar'])
                                a_index = pd.MultiIndex.from_product([list(a.loc[:,'Name'])], names=[Na])
                                J = pd.DataFrame(0, index=t_index, columns=a_index)
                                for ti in t.loc[:,'Name']:
                                    for ai in a.loc[:,'Name']:
                                         if ai.find(ti)!=-1: J.loc[ti,ai] = 1
                                J.loc['BEV','Exploiting BEV V2H for Electricity']=0
                                J.loc['BEV','Exploiting BEV V2H for Transport']=0
                                J.loc['BEV','Charging BEV V2H']=0
                                
                                # s, the market-share matrix which is by costruction an identity matrix
                                s = pd.DataFrame(0, index=a_index, columns=c_index)
                                for ai in range(len(a)):
                                    for ci in range(len(c)):
                                        if ai==ci: s.iloc[ai,ci] = 1
                                
                                i_index = pd.MultiIndex.from_product([list(i.loc[:,'Name'])], names=[Ni])
                                f_index = pd.MultiIndex.from_product([list(f.loc[:,'Name'])], names=[Nf])
                                l_index = pd.MultiIndex.from_product([list(l.loc[:,'Name'])], names=[Nl])
                                
                                
                            
                                "Reading matrices u,v,e,k,A,At"
                                u = pd.DataFrame(pd.read_excel(r"inputs/u - Support.xlsx", index_col=[0,1], header=[2]).values, index=n_index, columns=a_index)
                                A = pd.DataFrame(pd.read_excel(r"inputs/A - Support.xlsx", index_col=[0,1,2], header=[0]).values, index=a_index, columns=qo)
                                At = pd.DataFrame(pd.read_excel(r"inputs/At - Support.xlsx", index_col=[0,1], header=[0]).values, index=t_index, columns=qo)
                                k = pd.DataFrame(pd.read_excel(r"inputs/k - Support.xlsx", index_col=[0,1], header=[2]).values, index=t_index, columns=i_index)
                                # k_ = pd.DataFrame(pd.read_excel('{}/k_.xlsx'.format(pj_fld), index_col=[0,1], header=[2]).values, index=t_index, columns=iqi_index)
                                e = pd.DataFrame(pd.read_excel(r"inputs/e - Support.xlsx", index_col=[0,1], header=[2]).values, index=l_index, columns=a_index)
                                # e_ = pd.DataFrame(pd.read_excel('{}/e_.xlsx'.format(pj_fld), index_col=[0,1], header=[2]).values, index=l_index, columns=aqo_index)
                                v = pd.DataFrame(pd.read_excel(r"inputs/v - Support.xlsx", index_col=[0,1], header=[2]).values, index=f_index, columns=a_index)
                                # v_ = pd.DataFrame(pd.read_excel('{}/v_.xlsx'.format(pj_fld), index_col=[0,1], header=[2]).values, index=f_index, columns=aqo_index)
            
                                
                                "Modifying matrices"
                                # A
                                A_master = pd.read_excel("_master.xlsx", sheet_name="A",index_col=[0,1]).loc[location,:]
                                A.loc[A_master.index,:] = A_master.values
                                A.loc["Exploiting BEV for Transport",:] = bev_max_km
                                A.loc["Exploiting BEV V2H for Transport",:] = bev_v2h_max_km
                                A.loc["Exploiting ICEV for Transport",:] = icev_max_km
                                
                                # At
                                At_master = pd.read_excel("_master.xlsx", sheet_name="At",index_col=[0,1]).loc[location,:]
                                At.loc[At_master.index,:] = At_master.values
                                At.loc["ICEV",:] = icev_max_km
                                 
                                # Y
                                Y = pd.read_excel("_master.xlsx", sheet_name="Y",index_col=[0,1,2,3]).loc[(location,user,slice(None),[mileage,'-']),:]
                                Y.index = Y.index.get_level_values(2)
                                Y = Y.append(pd.DataFrame(np.zeros((1,Y.shape[1])), index=['Thermal coating'], columns=Y.columns))
                                Y= Y.reindex(["Electricity","Heating","Cooling","Hot Sanitary Water", "Transport","Cooking","Thermal coating"])
                               
                                # Y_coat
                                Y_coat = copy.deepcopy(Y)*0
                                Y_coat.loc[["Heating","Cooling"],:] =Y.loc[["Heating","Cooling"],:]*0.3
                                Y_coat = Y_coat.reindex(["Electricity","Heating","Cooling","Hot Sanitary Water", "Transport","Cooking","Thermal coating"])
                                
                                # e
                                e.loc["CO2","Exploiting ICEV for Transport"] = icev_emiss
                                e.loc["Charge from BEV ","Exploiting BEV for Transport"] = bev_cons
                                e.loc["Charge from BEV V2H ","Exploiting BEV V2H for Transport"] = bev_v2h_cons
                                
                                # v
                                v.loc["Costi operativi","Exploiting Grid for Electricity"] = ee_price
                                v.loc["Costi operativi","Exploiting ICEV for Transport"] = gas_price
                                
                                # k
                                k.loc["BEV","Costo di installazione"] = bev_price - bev_incentive
                                k.loc["ICEV","Costo di installazione"] = icev_price
                                k.loc["BEV V2H","Costo di installazione"] = bev_v2h_price - bev_incentive
                                k.loc["BEV","Capacità batteria"] = bev_capacity
                                k.loc["BEV V2H","Capacità batteria"] = bev_v2h_capacity
                                k.loc["Coat","Carbon footprint"] = 2*2.8*2.7*(house_surface/4 +4)
                                k.loc["Coat","Costo di installazione"] = 2*70*2.7*(house_surface/4 +4)*0.5#attivare solo per il caso senza incentivi
                                
                                
                                
                                "Building endogenous matrices"                            
                                D = pd.DataFrame(1, index=t_index, columns=['Installed units']) # Number of installed units
                                D_ = pd.DataFrame(1, index=t_index, columns=qi) # Number of installed units in different investment time-slices
                                C = pd.DataFrame(1, index=t_index, columns=['Installed capacity']) # Installed capacity
                                C_ = pd.DataFrame(1, index=t_index, columns=qi) # Installed capacity in different investment time-slices
                                SoC = pd.DataFrame(1, index=bat_index, columns=qo) # State of charge of every battery technology in every operational time-slice
                                X = pd.DataFrame(1, index=a_index, columns=qo) # Production of commodities in every operational time-slice
                                
                        
                        
                                "Generating optimization problem"
                                add_par = pd.read_excel(r'inputs\data_entry\Data_entry_{}.xlsx'.format(str(counter)), sheet_name='additional parameters', header=[0], index_col=[0,1])
                                dr = add_par.loc['Discount rate','Value'][0]
                                cp = add_par.loc['CO2 price','Value'][0]
                                
                                # Extracting slices of exogenous parameters
                                k_cost = k.loc[:,'Costo di installazione'].values
                                k_co2 = k.loc[:,'Carbon footprint'].values
                                e_co2 = e.loc['CO2',:].values
                                v_cost = v.loc['Costi operativi',:].values
                                stlv = k.loc[bat.loc[:,'Name'],'Capacità batteria'] # Upper limit of storage technologies
                                Tri = np.triu(np.ones((len(qo),len(qo)))) # Triangular matrix for SoC 
                                n_st = len(bat) # Number of storage technologies
                                grid_pos = list(t.loc[:,'Name']).index('Grid') # Position of grid
                                PVmax_pos = list(t.loc[:,'Name']).index('PV') # Position of PV
                                PV_pos = list(a.loc[:,'Name']).index('Exploiting PV for Electricity') # Position of PV
                                PV_limit = list(t.loc[:,'Name']).index('PV')#Position PV
                                Coat_pos = list(t.loc[:,'Name']).index('Coat') #Position of Coat
                                PS_pos = list(t.loc[:,'Name']).index('Psolar')#Position of solar panel
                                Boiler_pos = list(t.loc[:,'Name']).index('Gas boiler')#Position of Gas boiler
                                BEV_pos = list(t.loc[:,'Name']).index('BEV')#Position of BEV
                                BEV2H_pos = list(t.loc[:,'Name']).index('BEV V2H')#Position of BEV2H
                                ICEV_pos = list(t.loc[:,'Name']).index('ICEV')#Position of ICEV
                                Stove_pos = list(t.loc[:,'Name']).index('Gas Stove')#PPosition of gas stove
                                Ind_pos = list(t.loc[:,'Name']).index('Induction stove')#Position of induction stove
                                Heatpump_pos = list(t.loc[:,'Name']).index('Heat pump')#Position of Heat pump
                                Powerwall_pos = list(t.loc[:,'Name']).index('Powerwall')#Position of Powerwall
                                
    
    
    
                                D = cv.Variable(shape=D.shape, integer=True)
                                X = cv.Variable(shape=X.shape, nonneg=True)
                                Obj = cv.Minimize(cv.matmul((cp*k_co2+k_cost).T,D)+8760/len(qo)*sum(cv.sum(cv.sum(cv.matmul(cp*e_co2+v_cost,X),1,keepdims=True),1,keepdims=True)/(1+dr)**i for i in range(len(qi)))) # Miminization of costs
                                Obj_emi= cv.Minimize(sum(len(qi)*8760/len(qo)*(cv.sum(cv.matmul(e_co2,X), 1,keepdims=True)),(cv.sum(cv.matmul(k_co2.T, D), 1, keepdims= True))))
                                
                                constraints = [D[grid_pos] <= 1, # The grid contract can be 0 or 1
                                               D[Coat_pos]<= 1,#The Coat can be 0 or 1
                                               D[PV_limit] <= 20, # The number of PV 
                                               # X[PV_pos] == D[PV_pos]*A.values[PV_pos], # Production of electricity from PV equal to availability
                                               cv.matmul(G,cv.matmul(s.values.T,X)) == Y.values - D[Coat_pos]*Y_coat.values + cv.matmul(u.values,X), # Supply of need must be equal to demand and intermediate demand of need
                                               X <= cv.matmul(cv.diag(cv.matmul(J.T.values,D)),A.values), # Availability of activity
                                               cv.matmul(J.values,X) <= cv.matmul(cv.diag(D),At.values), # Availability of technology
                                               cv.matmul(X[-n_st:],Tri) - cv.matmul(cv.matmul(e,X)[-n_st:],Tri) <= cv.matmul(cv.diag(D[t.loc[t.Type=='Storage'].index]),np.repeat(stlv.values.reshape(stlv.shape), len(qo), 1)), # Cannot charge more than capacity of battery
                                               cv.matmul(X[-n_st:],Tri) - cv.matmul(cv.matmul(e,X)[-n_st:],Tri) >= 0.01*cv.matmul(cv.diag(D[t.loc[t.Type=='Storage'].index]),np.repeat(stlv.values.reshape(stlv.shape), len(qo), 1)) # Cannot run out of more than 1% of battery
                                               ]
                               
                                
                                "Solving model"
                                problem = cv.Problem(Obj, constraints)
                                problem.solve(verbose=True, solver='GUROBI')
                                
                                #tec_act = []
                                #act_nee = []
                                #act_tec = []
                                
                                #Xres_index = pd.MultiIndex.from_arrays([list(a.loc[:,'Name']),act_tec,act_nee], names=['Activity','Technology','Need'])
                                #Xres = pd.DataFrame(X.value, columns=qo, index=Xres_index)
                                Eres = pd.DataFrame(e.values@X.value, index=e.index, columns=qo)
                                Dres = pd.DataFrame(D.value,index=t_index, columns=['Installed units'])
                                # SOC = pd.DataFrame(X.value[-n_st:]@Tri - (e.values@X.value)[-n_st:]@Tri, index=bat_index, columns=qo)
                                # Yint = pd.DataFrame((u.values@Xres).values, index=n_index, columns=qo)
                                
                                #Xres_plot = Xres.T.stack([0,2]).fillna(0).loc[(slice(None),slice(None),['Electricity'])]
                                #Xres_plot.droplevel([0,1]).plot()
                                
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'Grid installation']= D.value[grid_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'Powerwall installation']= D.value[Powerwall_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'PV installation']= D.value[PV_limit]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'PS installation']= D.value[PS_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'Gas boiler installation']= D.value[Boiler_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'Heatpump installation']= D.value[Heatpump_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'BEV installation']= D.value[BEV_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'BEV V2H installation']= D.value[BEV2H_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'ICEV installation']= D.value[ICEV_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'Gas stove installation']= D.value[Stove_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'Induction stove installation']= D.value[Ind_pos]
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity,gas_price),'Coat installation']= D.value[Coat_pos]
                                #Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Car Model']= bev_model
                                
                                Grid_production=X.value[0]
                                PV_production=X.value[1]
                                HB_production=X.value[2]
                                BEVele_production=X.value[3]
                                BEV2Hele_production=X.value[4]
                                Boilerheat_production=X.value[5]
                                HPheat_production=X.value[6]
                                HPcool_production=X.value[7]
                                Boilerwater_production=X.value[8]
                                PS_production=X.value[9]
                                BEV_Km_production=X.value[10]
                                BEV2H_Km_production=X.value[11]
                                ICEV_Km_production=X.value[12]
                                GasStove_production=X.value[13]
                                IndStove_production=X.value[14]
                                Coat_production=X.value[15]
                                Sellsurplus_production=X.value[16]
                                ChargeHB_consumption=X.value[17]
                                ChargeBEV_consumption=X.value[18]
                                ChargeBEV2H_consumption=X.value[19]
                                
                                PV=np.sum(PV_production)
                                HB=np.sum(HB_production)
                                BEV2H=np.sum(BEV2Hele_production)
                                HP_H=np.sum(HPheat_production)
                                HP_C=np.sum(HPcool_production)
                                PS=np.sum(PS_production)
                                IND=np.sum(IndStove_production)
                                
                                NG=np.sum(Grid_production)
                                BO_H=np.sum(Boilerheat_production)
                                BO_W=np.sum(Boilerwater_production)
                                GA=np.sum(GasStove_production)
                                SS=np.sum(Sellsurplus_production)
                                
                                ShareGreen=(PV+HB+BEV2H+HP_H+HP_C+PS+IND)/(PV+HB+BEV2H+HP_H+HP_C+PS+IND+NG+BO_H+BO_W+GA)
                                electricitycost=(NG*ee_price + SS*0.05)/(NG+PV)
                                
                                costioperativi=np.dot(v_cost, X.value)
                                op_cost=np.sum(costioperativi)
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Operation Cost [€]']= op_cost
                                
                                I_cost=np.sum(k_cost.T@Dres.values)
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Investment Cost [€]']= I_cost
                                
                                tot_cost = op_cost*len(qi)*8760/len(qo)+I_cost
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Total Cost [€]']= tot_cost
                                
                                emiconsumi= np.dot(e_co2, X.value)
                                emi_op=np.sum(emiconsumi)
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Operation Emissions [KgCO2e/kWh]']= emi_op
                                
                                LCA=np.sum(k_co2.T@Dres.values)
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Life Cycle Assessment [KgCO2e]']= LCA
                                
                                tot_emi = emi_op*len(qi)*8760/len(qo)+LCA
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Total emissions [KgCO2e]']= tot_emi
                                
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Share Green Home']= ShareGreen
                                
                                Res.loc[(user,location, mileage,year,ee_price,bev_incentive,bev_capacity),'Electricity Price']= electricitycost
                                
                                database[user][location][mileage][year][ee_price][bev_incentive][bev_model] = Dres
                                
                                print(counter)
                                
        
                                Res.to_excel('Results/All_results_{counter}.xlsx')
        
                                counter += 1
                            

            
