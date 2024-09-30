#%%
import pypsa,os,requests,logging,tempfile,random , linopy, time, argparse, multiprocessing
import pandas as pd, numpy as np , matplotlib.pyplot as plt, xarray as xr

from pyomo.environ import Constraint
from multiprocessing import Pool

from windpowerlib import ModelChain, WindTurbine, WindFarm, create_power_curve,TurbineClusterModelChain, WindTurbineCluster
from windpowerlib import data as wt

import logging
logging.getLogger().setLevel(logging.DEBUG)

#a_file = pd.read_csv('../../models_inputs/RES generation data/PV generation data/pv_capacity_factor_hourly.csv')
#%%
def experiment_function(H2_selling_price_per_kg, simulation_horizon_number_of_years):
    #%%RES input data
    solar_load_factor_data = pd.read_csv('./Data/pv_cf.csv')
    wind_load_factor_data = pd.read_csv('./Data/wind_cf.csv')
    solar_load_factor_timeseries, wind_load_factor_timeseries = solar_load_factor_data['capacity_factor'], wind_load_factor_data['capacity_factor']

    #%%
    n_years = 10         #n_years is the number of years to which the csv parameters such as capex refer to
    simulation_years = simulation_horizon_number_of_years # number of simulation years. This parameter is inputed here

    solar_load_factor_timeseries_series, wind_load_factor_timeseries_series = solar_load_factor_timeseries, wind_load_factor_timeseries
    solar_load_factor_timeseries, wind_load_factor_timeseries = list(solar_load_factor_timeseries), list(wind_load_factor_timeseries)

    #%%Loads input data
    ng_demand_timeseries_data = pd.read_csv('./Data/athens_ng_demand.csv')
    ng_demand_timeseries =  round(ng_demand_timeseries_data['ng_demand(MWh)'] ,5) 
    ng_demand_timeseries_series = ng_demand_timeseries
    ng_demand_timeseries = list(ng_demand_timeseries)

    #Models Parameters input data
    input_parameters_data = pd.read_csv('./Data/input_parameters_S2.1.csv')

     #%%######################### NETWORK PARAMETERS #######################
    #Generators data
    wind_PPA_provider_capex, solar_PPA_provider_capex = input_parameters_data['wind_capex'][0] , input_parameters_data['solar_capex'][0]
    wind_fixed_opex, solar_fixed_opex = input_parameters_data['wind_fixed_opex'][0], input_parameters_data['solar_fixed_opex'][0]
    wind_PPA_provider_marginal, solar_PPA_provider_marginal = input_parameters_data['wind_marginal'][0], input_parameters_data['solar_marginal'][0]

    #H2 storage data
    H2_store_name  = 'H2 depot' 
    H2_storage_capex, H2_storage_marginal = input_parameters_data['H2_storage_capex'][0], input_parameters_data['H2_storage_marginal'][0]
    H2_storage_fixed_opex = input_parameters_data['H2_storage_fixed_opex'][0]
    e_nom_extendable, e_cyclic = True, False


    #NG generation data
    LHV_H2,HHV_H2 = 0.03333, 0.03989  #LHV,HHV of H2 in MWh/kg H2
    LHV_NG, HHV_NG = 0.0131, 0.01485  #LHV,HHV of NG in MWh/kg NG (a typical HHV ng is 14.49 kWh/kg)
    MHA = input_parameters_data['MHA'][0] #Max H2 admixture
    en_density_H2, en_density_ng = 3 , 10.167
    Mr_H2, Mr_ng = 2.01568 *1e-3, 17.47* 1e-3 # molar masses in kg per mol
    NG_marginal_cost = input_parameters_data['NG_marginal_cost'][0]

    #Selling prices data
    H2_sale_price_per_kg = H2_selling_price_per_kg
    H2_sale_price_per_MWh = H2_sale_price_per_kg / HHV_H2

    #Links data
    electrolysis_efficiency =  input_parameters_data['electrolysis_efficiency'][0]
    electrolysis_capex = input_parameters_data['electrolysis_capex'][0]
    electrolysis_fixed_opex = input_parameters_data['electrolysis_fixed_opex'][0]
    electrolysis_var_opex = input_parameters_data['electrolysis_variable_opex'][0]
    charge_efficiency, discharge_efficiency, H2_transport_efficiency = 0.99, 0.99, 0.99


    #Max H2 admixture
    #power_ratio =  round(en_density_H2/en_density_ng*MHA/(1-MHA),4)
    power_ratio = round(HHV_H2* Mr_H2/(HHV_NG*Mr_ng) * MHA/(1-MHA),4)

    #%%
    #Environmental/emissions parameters
    wind_generation_CO2_emissions_per_MWh, solar_generation_CO2_emissions_per_MWh = 10, 13
    GHG_emissions_per_MWh_NG = 2.75 /LHV_NG/1000 #GHG emissions in kg of CO2 per MWh of combusted NG (209.9 kg CO2 /MWh of comb.NG @ LHV)


    #Make 5Y or 10Y (or other horizon) simulation?
    if simulation_years != n_years:
        #Correct all components capexes:
        wind_PPA_provider_capex, solar_PPA_provider_capex = wind_PPA_provider_capex*simulation_years/10, solar_PPA_provider_capex*simulation_years/10
        electrolysis_capex, H2_storage_capex = electrolysis_capex*simulation_years/10 , H2_storage_capex*simulation_years/10
        
        #Correct wind and solar power timeseries lengths
        solar_load_factor_timeseries, wind_load_factor_timeseries = solar_load_factor_timeseries[:24*365*simulation_years], wind_load_factor_timeseries[:24*365*simulation_years]
        #Correct NG demand timeseries length
        ng_demand_timeseries = ng_demand_timeseries[:24*365*simulation_years] 



    #%% ############## NETWORK SETUP-PYPSA #############################
    ####################################################################
    network = pypsa.Network()
    network.set_snapshots(range(1, 24*365*simulation_years+1))

    #Add buses
    network.add("Bus", "Bus AC", carrier="AC")
    network.add("Bus", "Bus H2_1", carrier="H2")
    network.add("Bus", "Bus H2_2", carrier="H2")
    network.add("Bus", "Bus NG", carrier="NG")
        
    #Add carriers 
    #for carrier_name in carriers_names:
    #    network3.add("Carrier", carrier_name, co2_emissions = list(CO2_data[CO2_data['Carrier'] == carrier_name]['Emissions(t/Mwh of primary energy)'])[0])

    #Add loads
    network.add('Load', name= 'NG load', bus = 'Bus NG', p_set = ng_demand_timeseries)

    #Add Generators
    network.add(
            "Generator",
            name = "wind_provider_PPA",
            bus= 'Bus AC',
            carrier= 'AC',
            marginal_cost = wind_PPA_provider_marginal,
            p_nom_extendable= True , 
            p_max_pu= wind_load_factor_timeseries,
            capital_cost= wind_PPA_provider_capex,)

    network.add(
            "Generator",
            name = "solar_provider_PPA",
            bus= 'Bus AC',
            carrier= 'AC',
            marginal_cost = solar_PPA_provider_marginal,
            p_nom_extendable= True , 
            p_max_pu= solar_load_factor_timeseries,
            capital_cost=solar_PPA_provider_capex,)

    network.add(
            "Generator",
            name = "NG Generator",
            bus= 'Bus NG',
            carrier= 'NG',
            marginal_cost = NG_marginal_cost,
            p_nom_extendable= True , 
            capital_cost=0,)

    #Stores
    network.add("Store", name = "H2 depot", bus="Bus H2_2", e_cyclic=True, e_nom_extendable=True,
                marginal_cost = H2_storage_marginal,
                capital_cost  = H2_storage_capex)

    #Links
    network.add(
        "Link",
        "P_to_H2",
        bus0="Bus AC",
        bus1="Bus H2_1",
        efficiency= electrolysis_efficiency,
        capital_cost= electrolysis_capex,
        marginal_cost= electrolysis_var_opex,
        p_nom_extendable=True,) 
    
    network.add(
        "Link",
        "H2_charge",
        bus0="Bus H2_1",
        bus1="Bus H2_2",
        efficiency= charge_efficiency,
        capital_cost= 0,
        p_nom_extendable=True,)

    network.add(
        "Link",
        "H2_discharge",
        bus0="Bus H2_2",
        bus1="Bus H2_1",
        efficiency= discharge_efficiency,
        capital_cost= 0,
        p_nom_extendable=True,)

    network.add(
        "Link",
        "H2_to_NG",
        bus0="Bus H2_1",
        bus1="Bus NG",
        efficiency= H2_transport_efficiency,
        capital_cost= 0,
        marginal_cost= 0,  
        p_nom_extendable=True,) 


    #Create model
    model = network.optimize.create_model()



    #%%####### MODEL CORRECTIONS ################################
    # Add a constraint that guarantees maximum H2 admixture allowed when injecting H2 to the NG grid
    p_from_pure_NG = model.variables['Generator-p'].sel(Generator = 'NG Generator')
    p_from_H2 = model.variables['Link-p'].sel(Link = 'H2_to_NG') *H2_transport_efficiency
    model.add_constraints(p_from_H2   <= power_ratio*p_from_pure_NG ,name="Maximum H2 admixture",)


    #%% ####################### With NPV as objective function #################################
    discount_rate = 0.07
    #define investment frames
    InvPeriodFrames_list = []
    for year in range(1,simulation_years+1):
        #define investment frames
        start, end = 24*365*(year-1) +1 , 24*365*year+1
        investment_frame_range = range(start, end)
        InvPeriodFrames_list.append(investment_frame_range)


    #Compute Fixed Opex (per year) for generators, electrolyzer, H2 storage
    wind_fixed_opex_per_year =  wind_fixed_opex *  model.variables['Generator-p_nom'].loc['wind_provider_PPA']
    solar_fixed_opex_per_year = solar_fixed_opex * model.variables['Generator-p_nom'].loc['solar_provider_PPA']
    electrolyzer_fixed_opex_per_year = model.variables['Link-p_nom'].loc['P_to_H2'] * electrolysis_fixed_opex
    H2_storage_fixed_opex_per_year =  model.variables['Store-e_nom'] * H2_storage_fixed_opex
    total_fixed_opex_per_year =  wind_fixed_opex_per_year + solar_fixed_opex_per_year +electrolyzer_fixed_opex_per_year + H2_storage_fixed_opex_per_year 


    #Definition of NPV (obj.function will be minus the NPV). Capexes are payed in year zero. Minus because capex is an expense!
    objective_function_NPV  =       ( - model.variables['Generator-p_nom'].loc['wind_provider_PPA']* wind_PPA_provider_capex -   #wind  capex
            model.variables['Generator-p_nom'].loc['solar_provider_PPA']* solar_PPA_provider_capex - #solar capex
            model.variables['Link-p_nom'].loc['P_to_H2'] *network.links.T.P_to_H2['capital_cost'] - #electrolyzer capex
            model.variables['Store-e_nom'].loc['H2 depot'] * network.stores.loc['H2 depot', 'capital_cost'] #H2 storage capex
                                )


    #Compute cash flows of each year. 
    CFY1 = (model.variables['Link-p'].loc[InvPeriodFrames_list[0],'H2_to_NG'].sum() * H2_transport_efficiency* H2_sale_price_per_MWh - #H2 sales income 
            model.variables['Link-p'].loc[InvPeriodFrames_list[0],'P_to_H2'].sum()*network.links.T.P_to_H2['marginal_cost'] - #electrolyzer variable opex
            total_fixed_opex_per_year # fixed opex for Y1
        )

    objective_function_NPV += CFY1 /(1+discount_rate)
    #compute and add up the cash flows of the years >1. These years have only O&M costs.
    for year in range(2, simulation_years+1):
        ss_range = InvPeriodFrames_list[year-1]
        cash_flow_of_year = (model.variables['Link-p'].loc[ss_range,'H2_to_NG'].sum() * H2_transport_efficiency* H2_sale_price_per_MWh - #income from H2 sales
                            model.variables['Link-p'].loc[ss_range,'P_to_H2'].sum()*network.links.T.P_to_H2['marginal_cost']-  #electrolysis variable opex (water?)
                            total_fixed_opex_per_year #total fixed opex for that year
                            )

        objective_function_NPV+= cash_flow_of_year/((1+discount_rate)**year)   

    #Solve model with minus NPV as obj function.
    model.objective =   -objective_function_NPV

    #SOLVE
    experiment_start_time = time.perf_counter() #start measuring time of experiment
    network.optimize.solve_model()
    experiment_end_time = time.perf_counter() 
    experiment_duration  = round((experiment_end_time - experiment_start_time)/60/60,2) #experiment duration in hours


    ############ Useful quantities ##########################

    #network3.generators_t.p  #network3.generators.T
    #network3.links_t.p0      #network3.links.T
    #network3.loads_t.p       #network3.loads.T
    #network3.stores_t.p      #network3.stores.T

    #-network3.links_t.p1['H2_to_NG'].sum()*H2_transport_efficiency*H2_sale_price_per_MWh  #income of sold H2
    #network3.generators_t.p['wind_provider_PPA'].sum()*wind_PPA_provider_marginal #wind generation cost
    #network3.generators_t.p['solar_provider_PPA'].sum()*solar_PPA_provider_marginal  #solar generation cost



    # %% ############### COSTS BREAKDOWN & CO2 EMISSIONS ###################
    ########################################################################
    #Costs calculations
    #marginal (or variable opex) costs
    WF_production_costs = network.generators_t.p['wind_provider_PPA'].sum() * network.generators.loc['wind_provider_PPA','marginal_cost']
    SF_production_costs = network.generators_t.p['solar_provider_PPA'].sum() * network.generators.loc['solar_provider_PPA','marginal_cost']
    NGG_production_costs= network.generators_t.p['NG Generator'].sum() * network.generators.loc['NG Generator','marginal_cost']

    energy_production_costs =  WF_production_costs + SF_production_costs+ NGG_production_costs

    electrolysis_var_opex_costs = (network.links.T.loc['marginal_cost','P_to_H2']*network.links_t.p0.P_to_H2.sum() )
                            #network3.links.T.loc['marginal_cost','H2_to_NG']*network3.links_t.p0.H2_to_NG.sum() )
    H2_storage_var_opex_costs =  0                       

    variable_costs = energy_production_costs + electrolysis_var_opex_costs +H2_storage_var_opex_costs

    #capex costs
    capex_WF = network.generators.loc['wind_provider_PPA','capital_cost']*network.generators.loc['wind_provider_PPA','p_nom_opt']
    capex_SF = network.generators.loc['solar_provider_PPA','capital_cost']*network.generators.loc['solar_provider_PPA','p_nom_opt']
    capex_NGG = network.generators.loc['NG Generator','capital_cost']*network.generators.loc['NG Generator','p_nom_opt']
    capex_generators =  capex_WF + capex_SF

    capex_electrolyser = network.links.T.P_to_H2['p_nom_opt'] * network.links.T.P_to_H2['capital_cost']
    capex_H2_storage   = network.stores.loc['H2 depot','capital_cost']* network.stores.loc['H2 depot', 'e_nom_opt' ]
    capex_costs = capex_generators + capex_electrolyser + capex_H2_storage

    #Fixed OPEX costs total sim calculations
    fixed_opex_WF, fixed_opex_SF = wind_fixed_opex* network.generators.loc['wind_provider_PPA','p_nom_opt'] * simulation_years,  solar_fixed_opex* network.generators.loc['solar_provider_PPA','p_nom_opt'] * simulation_years
    fixed_opex_electrolysis = network.links.T.P_to_H2['p_nom_opt'] * electrolysis_fixed_opex *simulation_years
    fixed_opex_H2_storage = network.stores.loc['H2 depot', 'e_nom_opt' ] * H2_storage_fixed_opex *simulation_years
    fixed_opex_total =  fixed_opex_WF + fixed_opex_SF +fixed_opex_electrolysis +fixed_opex_H2_storage 

    opex_costs_total = variable_costs + fixed_opex_total
    costs_total = capex_costs + opex_costs_total

    #CO2 costs
    WF_CO2_costs = network.generators_t.p['wind_provider_PPA'].sum()*wind_generation_CO2_emissions_per_MWh
    SF_CO2_costs = network.generators_t.p['solar_provider_PPA'].sum()*solar_generation_CO2_emissions_per_MWh
    NGG_CO2_costs = network.generators_t.p['NG Generator'].sum()* GHG_emissions_per_MWh_NG
    power_generation_CO2_costs =  WF_CO2_costs + SF_CO2_costs 
    gas_burning_CO2_costs = NGG_CO2_costs
    total_CO2_costs = power_generation_CO2_costs + gas_burning_CO2_costs

    #CO2 emissions
    CO2_emissions_from_energy_production = (network.generators_t.p['wind_provider_PPA'].sum() * wind_generation_CO2_emissions_per_MWh +
                                            network.generators_t.p['solar_provider_PPA'].sum() * solar_generation_CO2_emissions_per_MWh)
    CO2_emissions_from_ng_burning = (network.generators_t.p['NG Generator'].sum() * GHG_emissions_per_MWh_NG )
    total_co2_emissions = CO2_emissions_from_energy_production + CO2_emissions_from_ng_burning
    total_co2_emissions = round(total_co2_emissions,2)

    #Power production calculations
    energy_from_wind_generation = network.generators_t.p['wind_provider_PPA'].sum()
    energy_from_solar_generation = network.generators_t.p['solar_provider_PPA'].sum()

    total_el_energy_production = energy_from_wind_generation + energy_from_solar_generation
    average_el_price_per_kWh = round((
                                energy_from_wind_generation/total_el_energy_production * network.generators.loc['wind_provider_PPA','marginal_cost'] +
                                energy_from_solar_generation/total_el_energy_production * network.generators.loc['solar_provider_PPA','marginal_cost'])/1000,3)

    #NG demand covered by synthetic H2 in P2G2 
    H2_content_of_NG = - network.links_t.p1['H2_to_NG'].sum()/network.loads_t.p_set['NG load'].sum()

    #=================PROFITABILITY FOR COMPANY CALCULATIONS (Expenses, Income, Profit) ===========
    #Costs and income for company
    H2_SALES_INCOMEv = - network.links_t.p1['H2_to_NG'].sum() *H2_sale_price_per_MWh 
    INCOMEv =  H2_SALES_INCOMEv
    net_income = INCOMEv - costs_total


    #Calculation of theoretical NPV (to make sure it is the same as objval)
    theoretical_NPV=0
    expenses_y1=0
    fixed_opex_WF_yearly, fixed_opex_SF_yearly = wind_fixed_opex* network.generators.loc['wind_provider_PPA','p_nom_opt'] ,  solar_fixed_opex* network.generators.loc['solar_provider_PPA','p_nom_opt'] 
    fixed_opex_electrolysis_yearly = network.links.T.P_to_H2['p_nom_opt'] * electrolysis_fixed_opex 
    fixed_opex_H2_storage_yearly = network.stores.loc['H2 depot', 'e_nom_opt' ] * H2_storage_fixed_opex
    fixed_opex_total_yearly = fixed_opex_WF_yearly + fixed_opex_SF_yearly + fixed_opex_electrolysis_yearly + fixed_opex_H2_storage_yearly

    #Year 0 (before investment becomes operational)
    expenses_y0 = capex_costs #capex is payed all in Y1
    theoretical_NPV-=expenses_y0 #minus because capex is an expense!

    #Year 1
    expenses_y1+= fixed_opex_total_yearly #fixed opex of Y1
    expenses_y1+= network.links.T.loc['marginal_cost','P_to_H2']*network.links_t.p0.P_to_H2[:365*24].sum() #electrolysis var opex
    income_y1 = - network.links_t.p1['H2_to_NG'][:365*24].sum() *H2_sale_price_per_MWh #sales income from selling H2
    cash_flow_y1 = income_y1 -expenses_y1
    theoretical_NPV+= cash_flow_y1/(1+discount_rate)

    #Next years
    for year in range(2, simulation_years+1):
        ss_range = InvPeriodFrames_list[year-1]
        cash_flow_of_year = ( - network.links_t.p1['H2_to_NG'][ss_range].sum() *H2_sale_price_per_MWh - #income from H2 sales
                                network.links.T.loc['marginal_cost','P_to_H2']*network.links_t.p0.P_to_H2[ss_range].sum()  - #electrolysis variable opex (water)
                                fixed_opex_total_yearly #total fixed opex for that year
                            )
    
        theoretical_NPV+= cash_flow_of_year/(1+discount_rate)**year

    #===============================================================================================
    #TECHNICAL statistics calculations
    wind_av_LF   = round(network.generators_t.p['wind_provider_PPA'].mean()/ (network.generators.loc['wind_provider_PPA','p_nom_opt']) ,4)
    solar_av_LF  = round(network.generators_t.p['solar_provider_PPA'].mean()/ (network.generators.loc['solar_provider_PPA','p_nom_opt']),4)
    electrolysis_av_LF = round(network.links_t.p0['P_to_H2'].mean()/ (network.links.T.P_to_H2['p_nom_opt']),4)
    H2_storage_capacity_kg = round(network.stores.loc['H2 depot', 'e_nom_opt' ]/LHV_H2 ,2)
    H2_storage_av_level = network.stores_t.e['H2 depot'].mean() / network.stores.loc['H2 depot', 'e_nom_opt' ]
    H2_to_NG_energies_av_ratio = (-network.links_t.p1['H2_to_NG']/network.generators_t.p['NG Generator']).mean() # avearge E_H2/H_NG
    H2_average_injection_ratio_per_volume = 1/(2+1/H2_to_NG_energies_av_ratio*(en_density_H2/en_density_ng))
    H2_total_production_in_tons = -network.links_t.p1['H2_to_NG'].sum()/HHV_H2/1000 #divided by ΗHV to obtain kg of H2, divided by 1000 to obtain tons
    H2_total_production_in_tons_per_year = H2_total_production_in_tons/ simulation_years

    #ENVIRONEMNTAL statistics calculations
    GHG_total_emissions_baseline = round(network.loads_t.p['NG load'].sum() * GHG_emissions_per_MWh_NG,3)   #tons of CO2 emitted in the case where 100% of NG demand is covered by NG
    GHG_total_emissions_scenario = round(network.generators_t.p['NG Generator'].sum() * GHG_emissions_per_MWh_NG,3) #tons of CO2 emitted under optimal solution
    GHG_emissions_fraction_of_baseline = round(GHG_total_emissions_scenario/GHG_total_emissions_baseline,4)

    #===============================================================================================
    theoretical_objval =  -round( theoretical_NPV ,2)#round(-(INCOMEv - EXPENSESv),2)#round(marginal_costs + capex_costs,2)
    model_objval = round(network.objective,2)

    #======================================================================================
    print('COSTS BREAKDOWN')
    print('CAPEX: '+' '*18,round(capex_costs,2),' €')
    #print('Generators capex (% of total system capex):'+' '*8,round(capex_generators/capex_costs*100,2), '%')
    print('Wind  capex (% of total system capex):',round(capex_WF/capex_costs*100,2), '%')
    print('Solar  capex (% of total system capex):',round(capex_SF/capex_costs*100,2), '%')
    print('Electrolyser capex (% of total system capex):', round(capex_electrolyser/capex_costs*100,2),'%' )
    print('H2 storage capex (% of total system capex):  ', round(capex_H2_storage/capex_costs*100,2), '%\n')

    print('OPEX: '+' '*17,round(opex_costs_total,2),' €')
    print('Wind  fixed opex (% of total system opex):'+' '*8,round(fixed_opex_WF/opex_costs_total*100,2), '%')
    print('Wind      v.opex (% of total system opex):',round(WF_production_costs/opex_costs_total*100,2), '%')
    print('Solar fixed opex (% of total system opex):'+' '*8,round(fixed_opex_SF/opex_costs_total*100,2), '%')
    print('Solar     v.opex (% of total system opex):',round(SF_production_costs/opex_costs_total*100,2), '%')
    print('Electrolysis fixed opex (% of total system opex):'+' '*8,round(fixed_opex_electrolysis/opex_costs_total*100,2), '%')
    print('Electrolysis v.opex(% of total system opex):',round(electrolysis_var_opex_costs/opex_costs_total*100,2),'%')
    print('H2 storage fixed opex (% of total system opex):'+' '*8,round(fixed_opex_H2_storage/opex_costs_total*100,2), '%\n')
    print('H2 storage v.opex (% of total system opex):'+' '*8,round(0/opex_costs_total*100,2), '%\n')
    print('==============================================')


    print('Theoretical objval  : ', round(theoretical_objval,2),' €')
    print('Model objval (-NPV) : ', round(model_objval,2), ' €')
    print('Difference (%): ', round((model_objval - theoretical_objval)/model_objval*100,4),'%')



    print('===========================================================','\n')
    print('COMPANY PROFITABILITY REPORT')
    print('Company horizon costs: ', round(costs_total,2), ' €')
    print('Capex (%): ', round(capex_costs/costs_total*100,2), '%')
    print('Opex (%): ', round(opex_costs_total/costs_total*100,2), '%')
    print('Company horizon income: ', round(INCOMEv,2), ' €')
    print('P2G(H2) income: ', round(H2_SALES_INCOMEv/INCOMEv*100,2),' %')
    print('Company horizon net profit: ', round(net_income,2),' €' )
    print('Return on Investment (ROI) (%): ',  round(net_income/capex_costs*100,2), ' %')


    print('==============================================')
    print('Power generation by wind energy(% of demand):', round(energy_from_wind_generation/total_el_energy_production*100,2),'%')
    print('Power generation by solar energy(% of demand):', round(energy_from_solar_generation/total_el_energy_production*100,2),'%')
    print('Average electricity prouction cost: ', average_el_price_per_kWh, ' €/kWh')

    print('\n =======================================')
    print('ECONOMIC STATISTICS')
    print('Investment NPV (should be zero): ',round(-model_objval,2) )
    print('H2 sale price : ', round(H2_sale_price_per_kg,2), '€/kg')

    print('\n =======================================')
    print('TECHNICAL')
    print('Wind nominal installation: ', round(network.generators.loc['wind_provider_PPA','p_nom_opt'],5), ' MW' )
    print('Wind av.capacity factor(% of p_nom): ', round(wind_av_LF*100,2) ,' %')
    print('Solar nominal installation: ', round(network.generators.loc['solar_provider_PPA','p_nom_opt'],5), ' MW' )
    print('Solar av.capacity factor(% of p_nom): ', round(solar_av_LF*100,2) ,' %')
    print('Electrolysis nominal installation: ', round(network.links.T.P_to_H2['p_nom_opt'],4), ' MW')
    print('Electrolysis av.capacity factor(% of p_nom): ', round(electrolysis_av_LF*100,5),' %')
    print('H2 storage size (kg): ',H2_storage_capacity_kg,' kg' )
    print('H2 storage av.storage level (%): ', round(H2_storage_av_level*100,2), ' %')
    print('H2 av.injection per volume (%): ', round(H2_average_injection_ratio_per_volume*100,4), ' %')
    print("H2 total production (tons): ", round(H2_total_production_in_tons,3))         
    print('NG energy demand covered by synthetic H2 (fraction):', round(H2_content_of_NG*100,2),'%\n')

    print('=======================================')
    print('ENVIRONMENTAL')
    print("GHG emissions of baseline:                 ", GHG_total_emissions_baseline, ' tons CO2 eq.')
    print("GHG emissions of scenario (% of baseline): ", round(GHG_emissions_fraction_of_baseline*100,2), ' %')
    print("GHG emissions savings: ", round(GHG_total_emissions_baseline - GHG_total_emissions_scenario,2), ' tons CO2 eq. \n')
    
    #%%################## WRITE RESULTS TO CSV #############################
    #df = pd.DataFrame()
    data = {'CAPEX(EUR)': [round(capex_costs,2)],'Wind capex (%)': [round(capex_WF/capex_costs*100,2)],'Solar capex (%)': [round(capex_SF/capex_costs*100,2)],'Electrolysis capex (%)': [round(capex_electrolyser/capex_costs*100,2)],'H2 storage capex (%)': [round(capex_H2_storage/capex_costs*100,2)],
            'OPEX(EUR)': round(opex_costs_total,2) , 'Wind Fix.Opex(%)':round(fixed_opex_WF/opex_costs_total*100,2) ,'Wind Var.Opex(%)': round(WF_production_costs/opex_costs_total*100,2),
            'Solar Fix.Opex(%)': round(fixed_opex_SF/opex_costs_total*100,2),'Solar Var.Opex(%)': round(SF_production_costs/opex_costs_total*100,2),
            'Electrolysis Fix.Opex(%)': round(fixed_opex_electrolysis/opex_costs_total*100,2), 'Electrolysis Var.Opex(%)': round(electrolysis_var_opex_costs/opex_costs_total*100,2),
            'H2 storage Fix.Opex(%)': round(fixed_opex_H2_storage/opex_costs_total*100,2) , 'H2 storage Var.Opex(%)': round(H2_storage_var_opex_costs/opex_costs_total*100,2) ,
            'Obj.val (-NPV EUR)': round(model_objval,2), 'Theoretical obj.val (-NPV EUR)': round(theoretical_objval,2), 'Difference (%) ': round((model_objval - theoretical_objval)/model_objval*100,4),
            'Company horizon costs(EUR)': round(costs_total,2), 'Capex(%)': round(capex_costs/costs_total*100,2),'Opex(%)':round(opex_costs_total/costs_total*100,2) , 'Company horizon income(EUR)':round(INCOMEv,2) , 'P2G(H2) income(%)':  round(H2_SALES_INCOMEv/INCOMEv*100,2),'Company horizon net profit(EUR)':round(net_income,2) ,
            'ROI(%)': round(net_income/capex_costs*100,2),
            'Wind generation(%)': round(energy_from_wind_generation/total_el_energy_production*100,2), 'Solar generation(%)': round(energy_from_solar_generation/total_el_energy_production*100,2),
            'Average electricity prod.cost(EUR/kWh)': average_el_price_per_kWh,'Investment NPV (should be zero)': round(-model_objval,2), 'H2 sale price(EUR/kg)': round(H2_sale_price_per_kg,2),
            'Wind nom.installation(MW)': round(network.generators.loc['wind_provider_PPA','p_nom_opt'],5), 'Wind av.capacity factor(% of p_nom)': round(wind_av_LF*100,2),
            'Solar nom.installation(MW)': round(network.generators.loc['solar_provider_PPA','p_nom_opt'],5), 'Solar av.capacity factor(% of p_nom)': round(solar_av_LF*100,2),
            'Electrolysis nominal installation(MW)': round(network.links.T.P_to_H2['p_nom_opt'],4) , 'Electrolysis av.capacity factor(% of p_nom)': round(electrolysis_av_LF*100,5),
            'H2 storage size (kg)': H2_storage_capacity_kg ,'H2 storage av.storage level (%)': round(H2_storage_av_level*100,2), 'H2 av.injection per volume (%)': round(H2_average_injection_ratio_per_volume*100,4) ,
            'H2 total production (tons):': round(H2_total_production_in_tons,3), 'H2 av. yearly production (tons):': round(H2_total_production_in_tons_per_year,3),
            'NG energy demand covered by synthetic H2 (%)':round(H2_content_of_NG*100,2) ,
            'GHG emissions of baseline (tons CO2 eq.)':GHG_total_emissions_baseline ,'GHG emissions of scenario (% of baseline)': round(GHG_emissions_fraction_of_baseline*100,2),'GHG emissions savings (tons CO2 eq.)': round(GHG_total_emissions_baseline - GHG_total_emissions_scenario,2),
            'Duration of experiment (h)': experiment_duration
            }

    # Save the results to csv
    df = pd.DataFrame(data)
    df = df.T
    #H2_sale_price_per_kg,H2_selling_price_per_kg =3.15, 3.15
    save_results_dir =  f'ATH_S1_{simulation_years}Y_H2_price_{H2_sale_price_per_kg}_EUR_per_kg'
    df.to_csv(save_results_dir)
    print(f'===========END OF EXPERIMENT WITH H2 SALE VALUE {H2_selling_price_per_kg}. ===================')
    

#%%Main function of the model. Uses argparse to put the "experiment function" into multiprocessing
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multiprocessing with argparse, with multiple H2 sale prices as parameters.")
    parser.add_argument('-v1',"--value1", type=float, default= 3.5, help="H2 price value ")
    parser.add_argument('-v2',"--value2", type=float, default= 4, help="H2 price value ")
    parser.add_argument('-v3',"--value3", type=float, default= 5, help="H2 price value ")
    parser.add_argument('-v4',"--value4", type=float, default= 6, help="H2 price value ")
    parser.add_argument('-v5',"--value5", type=float, default= 8, help="H2 price value ")
    parser.add_argument('-y',"--simulation_years", type=int, default= 1, help="Number of simulation horizon years (integer). From 1 to 10 ")
    args = parser.parse_args()
    H2_sales_prices_list = [args.value1, args.value2, args.value3,args.value4, args.value5]
    simulation_horizon_number_of_years = args.simulation_years

    # Create and start worker processes
    start_time = time.perf_counter()
    processes = []
    for H2_price in H2_sales_prices_list:
        p = multiprocessing.Process(target=experiment_function, args=(H2_price,simulation_horizon_number_of_years))
        processes.append(p)
        p.start()

    # Join all processes
    for p in processes:
        p.join()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Total time: {total_time}')


if __name__ == "__main__":
    main()

# %%
