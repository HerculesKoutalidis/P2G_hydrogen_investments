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
    project_lifetime = 25 #duration of the project horizon
    
    solar_load_factor_timeseries_series, wind_load_factor_timeseries_series = solar_load_factor_timeseries, wind_load_factor_timeseries
    solar_load_factor_timeseries, wind_load_factor_timeseries = list(solar_load_factor_timeseries), list(wind_load_factor_timeseries)

    #%%Loads input data
    ng_demand_timeseries_data = pd.read_csv('./Data/athens_ng_demand.csv')
    ng_demand_timeseries =  round(ng_demand_timeseries_data['ng_demand(MWh)'] ,5) 
    ng_demand_timeseries_series = ng_demand_timeseries
    ng_demand_timeseries = list(ng_demand_timeseries)

    #Models Parameters input data
    #input_parameters_data = pd.read_csv('./Data/input_parameters_S2.1.csv')

    #%%######################### NETWORK PARAMETERS #######################
    ####### INPUT EXPERIMENT PARAMETERS SECTION HERE ###########################
    #Fill the values of the parameters below, according to the "parameters guide xlsx" file
    wind_spec_capex = 1048800  
    wind_fixed_opex = 26030
    wind_marginal = 0
    solar_spec_capex =  646000
    solar_fixed_opex = 14487.5
    solar_marginal =  0
    H2_storage_spec_capex = 13775
    H2_storage_marginal = 0
    H2_storage_fixed_opex = 275.5
    NG_marginal_cost = 0
    electrolysis_efficiency =  0.79
    electrolysis_spec_capex = 877800  
    electrolysis_fixed_opex = 17556
    electrolysis_var_opex =   1.33 
    MHA =  0.2  #Max H2 admixture per volume ( 0 to 1) 
    sensitivity_analysis_scenario = 'LE1' 

    #############################################################################
    ########OTHER PARAMETERS (same for all experiments -DO NOT CHANGE)##########################
    #Generators data
    lifetime_wind, lifetime_solar = 27,30
    energy_basic_price = 0
    tax_on_energy = 0.06

    #H2 storage data
    H2_store_name  = 'H2 depot' 
    lifetime_storage = 20 #hydrogen storage lifetime
    e_nom_extendable, e_cyclic = True, False


    #NG generation data
    LHV_H2,HHV_H2 = 0.03333, 0.03989  #LHV,HHV of H2 in MWh/kg H2
    LHV_NG, HHV_NG = 0.0131, 0.01485  #LHV,HHV of NG in MWh/kg NG (a typical HHV ng is 14.49 kWh/kg)
    en_density_H2, en_density_ng = 3 , 10.167
    Mr_H2, Mr_ng = 2.01568 *1e-3, 17.47* 1e-3 # molar masses in kg per mol


    #Selling prices data
    H2_sale_price_per_kg = H2_selling_price_per_kg
    H2_sale_price_per_MWh = H2_sale_price_per_kg / HHV_H2

    #Links data
    electrolysis_TSO_cost_per_MW_per_year= 37464 #TSO transmission charge in €/MW/year
    tax_on_TSO_fee = 0.06 #tax on TSO fee (%)
    charge_efficiency, discharge_efficiency, H2_transport_efficiency = 0.99, 0.99, 0.99
    lifetime_electrolysis = 9 # electrolyzer stack lifetime
    
    #Balance of Plant (BoP) parameters
    BoP_sp_capex_pc = 0.02  #Percentage of electrolysis specific capex
    BoP_fix_opex_pc = 0.1  #Percentage of BoP fix opex per year, as a fraction of electrolysis specific fix opex
    BoP_electricity_consumption_pc = 0.01 #percentage of electrolysis electricity consumption
    electrolysis_spec_capex *= (1+BoP_sp_capex_pc) #include BoP sp.capex to electrolysis capex
    electrolysis_fixed_opex *= (1+BoP_fix_opex_pc)  #include BoP sp.fix opex to electrolysis fix.opex 
    electrolysis_efficiency /= (1+BoP_electricity_consumption_pc) #effective electrolysis efficiency.


    #Max H2 admixture
    #power_ratio =  round(en_density_H2/en_density_ng*MHA/(1-MHA),4)
    power_ratio = round(HHV_H2* Mr_H2/(HHV_NG*Mr_ng) * MHA/(1-MHA),4)
    
    #A string to recognize which model and sensitivity scenario produced the data in the csv export files
    recognition_string = 'wind_capex_'+str(wind_spec_capex)+'wind f.opex_'+str(wind_fixed_opex)+'_wind marginal_'+str(wind_marginal)
    recognition_string+= '_solar_capex_'+str(solar_spec_capex)+'solar f.opex_'+str(solar_fixed_opex)+'_solar marginal_'+str(solar_marginal)
    recognition_string+= '_H2_storage_capex_'+str(H2_storage_spec_capex)+'_H2_storage_marginal_'+str(H2_storage_marginal)+'_H2 storage f.opex_'+str(H2_storage_fixed_opex)
    recognition_string+= '_electr.eff_'+str(electrolysis_efficiency)+'_electr.capex_'+str(electrolysis_spec_capex)+'_electr.f.opex_'+str(electrolysis_fixed_opex)+'_electr. v.opex_'+str(electrolysis_var_opex)
    recognition_string+= '_MHA_'+str(MHA)

    if simulation_years != n_years:
        #Correct wind and solar power timeseries lengths
        solar_load_factor_timeseries, wind_load_factor_timeseries = solar_load_factor_timeseries[:24*365*simulation_years], wind_load_factor_timeseries[:24*365*simulation_years]
        #Correct NG demand timeseries length
        ng_demand_timeseries = ng_demand_timeseries[:24*365*simulation_years] 


    #Environmental/emissions parameters
    wind_generation_CO2_emissions_per_MWh, solar_generation_CO2_emissions_per_MWh = 10, 13
    GHG_emissions_per_MWh_NG = 2.75 /HHV_NG/1000 #GHG emissions in kg of CO2 per MWh of combusted NG (209.9 kg CO2 /MWh of comb.NG @ LHV)


    
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
            marginal_cost = wind_marginal,
            p_nom_extendable= True , 
            p_max_pu= wind_load_factor_timeseries,
            capital_cost= wind_spec_capex,)

    network.add(
            "Generator",
            name = "solar_provider_PPA",
            bus= 'Bus AC',
            carrier= 'AC',
            marginal_cost = solar_marginal,
            p_nom_extendable= True , 
            p_max_pu= solar_load_factor_timeseries,
            capital_cost=solar_spec_capex,)

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
                capital_cost  = H2_storage_spec_capex)

    #Links
    network.add(
        "Link",
        "P_to_H2",
        bus0="Bus AC",
        bus1="Bus H2_1",
        efficiency= electrolysis_efficiency,
        capital_cost= electrolysis_spec_capex,
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



    #%% ####################### With annualized costs and income objective function #################################
    discount_rate = 0.07
    def disc_factor(discount_rate, years): return ((1+discount_rate)**years)
    def CapitalRecoveryFactor(discount_rate, years): return round( discount_rate*(1+discount_rate)**years/((1+discount_rate)**years-1) ,4)
    def project_lifetime_mult_factor(discount_rate, project_lifetime):
        multiplication_factor = 0
        for year in range(1, project_lifetime+1):
            multiplication_factor += 1/disc_factor(discount_rate, year)
        return     multiplication_factor

    CRF = CapitalRecoveryFactor(discount_rate, project_lifetime)
    PLMF = project_lifetime_mult_factor(discount_rate, project_lifetime)

    #%%
    #define investment frames
    InvPeriodFrames_list = []
    for year in range(1,simulation_years+1):
        #define investment frames
        start, end = 24*365*(year-1) +1 , 24*365*year+1
        investment_frame_range = range(start, end)
        InvPeriodFrames_list.append(investment_frame_range)

    
    #Compute annualized Capex (payed per year) for generators, electrolyzer, H2 storage, (compression)
    Cinv_wind, Cinv_solar =  model.variables['Generator-p_nom'].loc['wind_provider_PPA']* wind_spec_capex ,  model.variables['Generator-p_nom'].loc['solar_provider_PPA']* solar_spec_capex
    capex_ann_wind ,capex_ann_solar =  CRF * Cinv_wind, CRF * Cinv_solar
    
    Cinv_electrolysis = model.variables['Link-p_nom'].loc['P_to_H2'] *network.links.T.P_to_H2['capital_cost'] #electrolyzer capex
    capex_ann_electrolysis = CRF * Cinv_electrolysis

    Cinv_storage = model.variables['Store-e_nom'].loc['H2 depot'] * network.stores.loc['H2 depot', 'capital_cost'] #H2 storage capex
    capex_ann_storage = CRF * Cinv_storage

    Cinv_ann =  capex_ann_wind + capex_ann_solar + capex_ann_electrolysis + capex_ann_storage


    #Compute Fixed Opex (per year) for generators, electrolyzer, H2 storage
    wind_fixed_opex_per_year =  wind_fixed_opex *  model.variables['Generator-p_nom'].loc['wind_provider_PPA']
    solar_fixed_opex_per_year = solar_fixed_opex * model.variables['Generator-p_nom'].loc['solar_provider_PPA']
    electrolyzer_fixed_opex_per_year =  (electrolysis_fixed_opex +electrolysis_TSO_cost_per_MW_per_year*(1+tax_on_TSO_fee)) *model.variables['Link-p_nom'].loc['P_to_H2'] 
    H2_storage_fixed_opex_per_year =  model.variables['Store-e_nom'] * H2_storage_fixed_opex
    Cfix_opex_ann =  wind_fixed_opex_per_year + solar_fixed_opex_per_year +electrolyzer_fixed_opex_per_year + H2_storage_fixed_opex_per_year 
    
    #Compute annualized Replacement incomes for electrolyzer, H2 storage, (compression) (wind and solar have lifetimes > project, so they are not replaced at all)
    Crep_electro, Crep_storage = 0.3 * Cinv_electrolysis , 0.75 * Cinv_storage
    Crep_electro_ann = CRF/disc_factor(discount_rate,lifetime_electrolysis) * Crep_electro + CRF/disc_factor(discount_rate,2*lifetime_electrolysis) * Crep_electro #electrolysis stack is replaced at years 9 and 18
    Crep_storage_ann = CRF/disc_factor(discount_rate,lifetime_storage) * Crep_storage
    Crep_ann = Crep_electro_ann + Crep_storage_ann


    #Compute annualized salvage income from selling components at the end of the project (wind, solar, electrolyzer, storage)
    Inc_salv_wind , Inc_salv_solar = 0.074 *(0.7 * Cinv_wind) , 0.2* (0.7*Cinv_solar) 
    Inc_salv_wind_ann, Inc_salv_solar_ann = CRF/disc_factor(discount_rate,project_lifetime) * Inc_salv_wind, CRF/disc_factor(discount_rate,project_lifetime) * Inc_salv_solar
    Inc_salv_electro, Inc_salv_storage = 0.222* Crep_electro, 0.75 * Crep_storage
    Inc_salv_electro_ann, Inc_salv_storage_ann = CRF/disc_factor(discount_rate,project_lifetime) *Inc_salv_electro, CRF/disc_factor(discount_rate,project_lifetime) *Inc_salv_storage
    Inc_salv_ann = Inc_salv_wind_ann + Inc_salv_solar_ann + Inc_salv_electro_ann + Inc_salv_storage_ann



    ############## DEFINITION OF OBJECTIVE FUNCTION ######################################################################################################
    #Definition of NPV (obj.function will be minus the NPV). Minus because capex ann and fixed costs ann are expenses!
    #Income from hydrogen sale. 
    Inc_from_H2_year = model.variables['Link-p'].loc[InvPeriodFrames_list[0],'H2_to_NG'].sum() * H2_transport_efficiency* H2_sale_price_per_MWh #H2 sales income FOR Y1
    
    #Compute Variable Opexes
    Cvaropex_electrolysis_year = (model.variables['Link-p'].loc[InvPeriodFrames_list[0],'P_to_H2'].sum()*network.links.T.P_to_H2['marginal_cost'] + #electrolyzer variable opex other than electricity and taxes on it        
                                  model.variables['Link-p'].loc[InvPeriodFrames_list[0],'P_to_H2'].sum()*(1+BoP_electricity_consumption_pc) * energy_basic_price* tax_on_energy)     #tax on electricity consumption - to the state                     
    Cvaropex_storage_year  = 0 
    Cvar_opex_year = Cvaropex_electrolysis_year + Cvaropex_storage_year

    objective_function_model = Inc_from_H2_year + Inc_salv_ann -( Cinv_ann + Cfix_opex_ann + Crep_ann*0) - Cvar_opex_year   #Cash flow of Y1
        
    #compute and add up the cash flows of the years >1.
    for year in range(2, simulation_years+1):
        ss_range = InvPeriodFrames_list[year-1]

        #Compute the Var.opexes for this year
        Cvaropex_electrolysis_year = (model.variables['Link-p'].loc[InvPeriodFrames_list[year-1],'P_to_H2'].sum()*network.links.T.P_to_H2['marginal_cost'] + #electrolyzer variable opex other than electricity and taxes on it        
                                      model.variables['Link-p'].loc[InvPeriodFrames_list[year-1],'P_to_H2'].sum() *(1+BoP_electricity_consumption_pc) * energy_basic_price* tax_on_energy)     #tax on electricity consumption - to the state                     
        Cvaropex_storage_year  = 0 
        Cvar_opex_year = Cvaropex_electrolysis_year + Cvaropex_storage_year

        #Compute H2 income for this year
        Inc_from_H2_year = model.variables['Link-p'].loc[InvPeriodFrames_list[year-1],'H2_to_NG'].sum() * H2_transport_efficiency* H2_sale_price_per_MWh #H2 sales income FOR Y1

        cash_flow_of_year = Inc_from_H2_year + Inc_salv_ann -( Cinv_ann + Cfix_opex_ann + Crep_ann) - Cvar_opex_year   #Cash flow of that year                    
        objective_function_model+= cash_flow_of_year  

    #Obj.function definition (minus ann. cash flow)
    model.objective =   -objective_function_model

    #SOLVE
    experiment_start_time = time.perf_counter() #start measuring time of experiment
    network.optimize.solve_model()
    experiment_end_time = time.perf_counter() 
    experiment_duration  = round((experiment_end_time - experiment_start_time)/60/60,2) #experiment duration in hours




    #%%Calculation of theoretical Objective  (to make sure it is the same as objval)
    #Year 1
    #Theoretical Income from selling H2 Y1
    Inc_from_H2_year_th = - network.links_t.p1['H2_to_NG'][:365*24].sum() *H2_sale_price_per_MWh

    #theoretical capexes calculations
    Cinv_wind_th  = network.generators.loc['wind_provider_PPA','capital_cost']*network.generators.loc['wind_provider_PPA','p_nom_opt']
    Cinv_solar_th  = network.generators.loc['solar_provider_PPA','capital_cost']*network.generators.loc['solar_provider_PPA','p_nom_opt']
    capex_ann_wind_th ,capex_ann_solar_th = CRF * Cinv_wind_th, CRF * Cinv_solar_th

    Cinv_electrolysis_th = network.links.T.P_to_H2['p_nom_opt'] * network.links.T.P_to_H2['capital_cost']
    capex_ann_electrolysis_th = CRF * Cinv_electrolysis_th
    
    Cinv_storage_th = network.stores.loc['H2 depot','capital_cost']* network.stores.loc['H2 depot', 'e_nom_opt' ]
    capex_ann_storage_th  = CRF * Cinv_storage_th
    Cinv_ann_th =  capex_ann_wind_th + capex_ann_solar_th + capex_ann_electrolysis_th + capex_ann_storage_th

    #Theoretical fix. opexes calculation
    wind_fixed_opex_per_year_th = wind_fixed_opex   * network.generators.loc['wind_provider_PPA','p_nom_opt']
    solar_fixed_opex_per_year_th = solar_fixed_opex * network.generators.loc['solar_provider_PPA','p_nom_opt']
    electrolyzer_fixed_opex_per_year_th = (electrolysis_fixed_opex +electrolysis_TSO_cost_per_MW_per_year*(1+tax_on_TSO_fee)) * network.links.T.P_to_H2['p_nom_opt']
    H2_storage_fixed_opex_per_year_th =  H2_storage_fixed_opex *  network.stores.loc['H2 depot', 'e_nom_opt' ]
    Cfix_opex_ann_th = wind_fixed_opex_per_year_th + solar_fixed_opex_per_year_th + electrolyzer_fixed_opex_per_year_th + H2_storage_fixed_opex_per_year_th
    
    #Theoretical annualized Replacement costs calculations (only electrolysis , H2 storage, (compression))
    Crep_electro_th, Crep_storage_th = 0.3 * Cinv_electrolysis_th , 0.75 * Cinv_storage_th
    Crep_electro_ann_th = CRF * Crep_electro_th/disc_factor(discount_rate,lifetime_electrolysis) + CRF * Crep_electro_th/disc_factor(discount_rate,2*lifetime_electrolysis) #electrolysis stack is replaced at years 9 and 18
    Crep_storage_ann_th = CRF * Crep_storage_th /disc_factor(discount_rate,lifetime_storage)
    Crep_ann_th = Crep_electro_ann_th + Crep_storage_ann_th

    #Theoretical annualized salvage income from selling components at the end of the project (wind, solar, electrolyzer, storage)
    Inc_salv_wind_th , Inc_salv_solar_th = 0.074 *(0.7 * Cinv_wind_th) , 0.2* (0.7*Cinv_solar_th) 
    Inc_salv_wind_ann_th, Inc_salv_solar_ann_th = CRF * Inc_salv_wind_th/disc_factor(discount_rate,project_lifetime), CRF * Inc_salv_solar_th/disc_factor(discount_rate,project_lifetime)
    Inc_salv_electro_th, Inc_salv_storage_th = 0.222* Crep_electro_th, 0.75 * Crep_storage_th
    Inc_salv_electro_ann_th, Inc_salv_storage_ann_th = CRF*Inc_salv_electro_th /disc_factor(discount_rate,project_lifetime) , CRF*Inc_salv_storage_th/disc_factor(discount_rate,project_lifetime) 
    Inc_salv_ann_th = Inc_salv_wind_ann_th + Inc_salv_solar_ann_th + Inc_salv_electro_ann_th + Inc_salv_storage_ann_th
    
    #Theoretical Var.opexes calculations
    Cvaropex_electrolysis_year_th = (network.links_t.p0['P_to_H2'][:365*24].sum()*network.links.T.P_to_H2['marginal_cost'] + #electrolyzer variable opex other than electricity and taxes on it
                                     network.links_t.p0['P_to_H2'][:365*24].sum() *(1+BoP_electricity_consumption_pc) *energy_basic_price* tax_on_energy)     #tax on electricity consumption - to the state 
    Cvaropex_storage_year_th  = 0 
    Cvar_opex_year_th = Cvaropex_electrolysis_year_th + Cvaropex_storage_year_th
    
    objective_function_th = Inc_from_H2_year_th + Inc_salv_ann_th -(Cinv_ann_th + Cfix_opex_ann_th + Crep_ann_th*0) - Cvar_opex_year_th

    #Next years
    for year in range(2, simulation_years+1):
        ss_range = InvPeriodFrames_list[year-1]
        
        #Theor. income from H2 sales this year
        Inc_from_H2_year_th = - network.links_t.p1['H2_to_NG'][ss_range].sum() *H2_sale_price_per_MWh

        #Theor var.opexes this year
        Cvaropex_electrolysis_year_th = (network.links_t.p0['P_to_H2'][ss_range].sum()*network.links.T.P_to_H2['marginal_cost'] + #electrolyzer variable opex other than electricity and taxes on it
                                     network.links_t.p0['P_to_H2'][ss_range].sum() *(1+BoP_electricity_consumption_pc) *energy_basic_price* tax_on_energy)     #tax on electricity consumption - to the state 
        Cvaropex_storage_year_th  = 0 
        Cvar_opex_year_th = Cvaropex_electrolysis_year_th + Cvaropex_storage_year_th
        
        objective_function_th += Inc_from_H2_year_th + Inc_salv_ann_th -(Cinv_ann_th + Cfix_opex_ann_th + Crep_ann_th) - Cvar_opex_year_th


    # %% ############### COSTS BREAKDOWN & CO2 EMISSIONS ###################
    ########################################################################
    #Costs calculations
    #capex costs
    capex_WF_project = network.generators.loc['wind_provider_PPA','capital_cost']*network.generators.loc['wind_provider_PPA','p_nom_opt']
    capex_SF_project = network.generators.loc['solar_provider_PPA','capital_cost']*network.generators.loc['solar_provider_PPA','p_nom_opt']
    capex_NGG_project = network.generators.loc['NG Generator','capital_cost']*network.generators.loc['NG Generator','p_nom_opt']
    capex_generators_project =  capex_WF_project + capex_SF_project

    capex_electrolyzer_project = network.links.T.P_to_H2['p_nom_opt'] * network.links.T.P_to_H2['capital_cost']
    capex_H2_storage_project   = network.stores.loc['H2 depot','capital_cost']* network.stores.loc['H2 depot', 'e_nom_opt' ]
    capex_costs_project = capex_generators_project + capex_electrolyzer_project + capex_H2_storage_project
    capex_ann = capex_costs_project *CRF

    #Fixed OPEX costs total sim, and project calculations
    fixed_opex_WF_ann, fixed_opex_SF_ann = wind_fixed_opex* network.generators.loc['wind_provider_PPA','p_nom_opt'] ,  solar_fixed_opex* network.generators.loc['solar_provider_PPA','p_nom_opt'] 
    fixed_opex_WF_project, fixed_opex_SF_project = fixed_opex_WF_ann * PLMF, fixed_opex_SF_ann * PLMF
    fixed_opex_electrolysis_ann = (electrolysis_fixed_opex +electrolysis_TSO_cost_per_MW_per_year*(1+tax_on_TSO_fee)) * network.links.T.P_to_H2['p_nom_opt'] 
    fixed_opex_electrolysis_project = fixed_opex_electrolysis_ann *PLMF 
    fixed_opex_H2_storage_ann = network.stores.loc['H2 depot', 'e_nom_opt' ] * H2_storage_fixed_opex 
    fixed_opex_H2_storage_project = fixed_opex_H2_storage_ann * PLMF
    fixed_opex_ann =  fixed_opex_WF_ann + fixed_opex_SF_ann +fixed_opex_electrolysis_ann +fixed_opex_H2_storage_ann 
    fixed_opex_project = fixed_opex_ann * PLMF

    #Var opex costs simulation total
    wind_var_opex_sim, solar_var_opex_sim =  0,0 
    wind_var_opex_ann, solar_var_opex_ann = wind_var_opex_sim/simulation_years , solar_var_opex_sim/simulation_years
    wind_var_opex_project, solar_var_opex_project = wind_var_opex_ann * PLMF, solar_var_opex_ann * PLMF
    NGG_var_opex_sim= network.generators_t.p['NG Generator'].sum() * network.generators.loc['NG Generator','marginal_cost']
    NGG_var_opex_sim_ann = NGG_var_opex_sim/simulation_years
    NGG_var_opex_sim_project = NGG_var_opex_sim_ann * PLMF
    energy_production_costs_sim =  wind_var_opex_sim + solar_var_opex_sim
    electrolysis_var_opex_sim = (network.links.T.loc['marginal_cost','P_to_H2']*network.links_t.p0.P_to_H2.sum() + # electrolysis var.opex other than electricity related)
                                 network.links_t.p0.P_to_H2.sum() *  energy_basic_price* tax_on_energy) #tax on electricity consumption)
    electrolysis_var_opex_ann = electrolysis_var_opex_sim/simulation_years
    electrolysis_var_opex_project = electrolysis_var_opex_ann * PLMF
    H2_storage_var_opex_costs_sim =  0
    H2_storage_var_opex_costs_ann =     H2_storage_var_opex_costs_sim/simulation_years    
    H2_storage_var_opex_costs_project = H2_storage_var_opex_costs_ann * PLMF            
    var_opex_sim = energy_production_costs_sim + electrolysis_var_opex_sim +H2_storage_var_opex_costs_sim
    var_opex_ann = var_opex_sim/simulation_years   #average annualized var.opex (all components) based on the sim
    var_opex_project = var_opex_ann * PLMF
    
    #Total opex calcs.
    opex_costs_ann = fixed_opex_ann + var_opex_ann
    opex_costs_nom_total = opex_costs_ann * project_lifetime #(?)
    opex_costs_project = opex_costs_ann *PLMF
    
    #Replacement costs calculations
    rep_cost_electrolyzer, rep_cost_storage =  0.3* capex_electrolyzer_project , 0.75 * capex_H2_storage_project
    rep_costs_electrolyzer_ann =  CRF * rep_cost_electrolyzer/disc_factor(discount_rate,lifetime_electrolysis) + CRF * rep_cost_electrolyzer/disc_factor(discount_rate,2*lifetime_electrolysis)
    rep_costs_electrolyzer_project = rep_costs_electrolyzer_ann * PLMF
    rep_costs_storage_ann = CRF * rep_cost_storage /disc_factor(discount_rate,lifetime_storage)
    rep_costs_storage_project = rep_costs_storage_ann * PLMF
    rep_costs_ann = rep_costs_electrolyzer_ann + rep_costs_storage_ann
    rep_costs_project = rep_costs_ann *PLMF

    #Total costs calcs.
    costs_project =  capex_costs_project + opex_costs_project + rep_costs_project
    costs_ann = costs_project * CRF

    #Salvage Income calculations
    salv_inc_wind_ann, salv_inc_solar_ann = Inc_salv_wind_ann_th, Inc_salv_solar_ann_th
    salv_inc_wind_project, salv_inc_solar_project = salv_inc_wind_ann * PLMF, salv_inc_solar_ann *PLMF
    salv_inc_electro_ann, salv_inc_storage_ann = Inc_salv_electro_th , Inc_salv_storage_th
    salv_inc_electro_project, salv_inc_storage_project = salv_inc_electro_ann* PLMF, salv_inc_storage_ann* PLMF
    salv_inc_ann = salv_inc_wind_ann + salv_inc_solar_ann + salv_inc_electro_ann + salv_inc_storage_ann
    salv_inc_project = salv_inc_ann * PLMF

    #H2 sales Income calculations
    h2_income_ann =  - network.links_t.p1['H2_to_NG'].sum() *H2_sale_price_per_MWh /simulation_years #total H2 income from simulation div by sim years
    h2_income_project = h2_income_ann * PLMF


    #Total income calcs.
    income_ann =  h2_income_ann + salv_inc_ann
    income_nom_total = income_ann * project_lifetime
    income_project = income_ann *PLMF

    #Net profit calcs.
    cash_flow_ann = income_ann - costs_ann
    NPV = cash_flow_ann * PLMF  #(?)
    Net_nom_profit_project =  income_nom_total - capex_costs_project - opex_costs_nom_total


    #Other calculations--------------------------------------------------------------------------------------------
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
    GHG_total_emissions_baseline_combustion = round(2.75*network.loads_t.p['NG load'].sum()/HHV_NG/1000,3)   #tons of CO2 emitted in the case where 100% of NG demand is covered by NG
    GHG_total_emissions_baseline_from_losses = round( 0.2828*network.loads_t.p['NG load'].sum()/HHV_NG/1000,3) #tons of CO2 eq. emitted due to NG leakage of 1%
    GHG_total_emissions_baseline = GHG_total_emissions_baseline_combustion + GHG_total_emissions_baseline_from_losses
    GHG_total_emissions_baseline_yearly = round(GHG_total_emissions_baseline/simulation_years,3) #tons yearly

    GHG_total_emissions_scenario_combustion = round(2.75*network.generators_t.p['NG Generator'].sum()/HHV_NG/1000,3) #tons of CO2 emitted from combustion of NG under optimal solution
    GHG_total_emissions_scenario_from_losses = round(0.2828*network.generators_t.p['NG Generator'].sum()/HHV_NG/1000,3) #tons of Co2 eq. due to NG leakage
    GHG_total_emissions_scenario = GHG_total_emissions_scenario_combustion + GHG_total_emissions_scenario_from_losses
    GHG_total_emissions_scenario_yearly = round(GHG_total_emissions_scenario/simulation_years,3) #tons yearly
    GHG_emissions_fraction_of_baseline = round(GHG_total_emissions_scenario/GHG_total_emissions_baseline,4)


    #=============== LCOH calculation =========================================================== 
    H2_production_ann = H2_total_production_in_tons / simulation_years
    H2_production_project = H2_production_ann * PLMF
    LCOH =   costs_project/(H2_production_project  * 1000) #mult. by 1000 to obtain kg
    
    #===============================================================================================
    theoretical_objval =  -round( objective_function_th ,6)
    model_objval = round(network.objective,6)

    #======================================================================================
    print('Duration of experiment (h)', experiment_duration)
    print('COSTS BREAKDOWN')
    print('CAPEX: '+' '*18,round(capex_costs_project,2),' €')
    #print('Generators capex (% of total system capex):'+' '*8,round(capex_generators/capex_costs*100,2), '%')
    print('Wind  capex (% of total system capex):',round(capex_WF_project/capex_costs_project*100,2), '%')
    print('Solar  capex (% of total system capex):',round(capex_SF_project/capex_costs_project*100,2), '%')
    print('Electrolyser capex (% of total system capex):', round(capex_electrolyzer_project/capex_costs_project*100,2),'%' )
    print('H2 storage capex (% of total system capex):  ', round(capex_H2_storage_project/capex_costs_project*100,2), '%\n')

    print('OPEX (PV): '+' '*17,round(opex_costs_project,2),' €')
    print('Wind  fixed opex (% of total system opex):'+' '*8,round(fixed_opex_WF_project/opex_costs_project*100,2), '%')
    print('Wind      v.opex (% of total system opex):',round(wind_var_opex_project/opex_costs_project*100,2), '%')
    print('Solar fixed opex (% of total system opex):'+' '*8,round(fixed_opex_SF_project/opex_costs_project*100,2), '%')
    print('Solar     v.opex (% of total system opex):',round(solar_var_opex_project/opex_costs_project*100,2), '%')
    print('Electrolysis fixed opex (% of total system opex):'+' '*8,round(fixed_opex_electrolysis_project/opex_costs_project*100,2), '%')
    print('Electrolysis v.opex(% of total system opex):',round(electrolysis_var_opex_project/opex_costs_project*100,2),'%')
    print('H2 storage fixed opex (% of total system opex):'+' '*8,round(fixed_opex_H2_storage_project/opex_costs_project*100,2), '%')
    print('H2 storage v.opex (% of total system opex):'+' '*8,round(H2_storage_var_opex_costs_project/opex_costs_project*100,2), '%\n')

    print('Replacement(PV): '+' '*13, round(rep_costs_project,2),' €' )
    print('Electrolysis repl.costs(% of total system repl.): ', round(rep_costs_electrolyzer_project/rep_costs_project,2), '%' )
    print('H2 storage repl.costs(% of total system repl.): ', round(rep_costs_storage_project/rep_costs_project,2), '%' )

    print('==============================================')


    print('Theoretical objval  : ', round(theoretical_objval,2),' €')
    print('Model objval : ', round(model_objval,2), ' €')
    print('Difference (%): ', round((model_objval - theoretical_objval)/model_objval*100,4),'%')



    print('===========================================================','\n')
    print('COMPANY PROFITABILITY REPORT')
    print('Company horizon costs (PV): ', round(costs_project,2), ' €')
    print('Capex (%): ', round(capex_costs_project/costs_project*100,2), '%')
    print('Opex (%): ', round(opex_costs_project/costs_project*100,2), '%')
    print('Replacement (%): ', round(rep_costs_project/costs_project*100,2), '%')
    print('Company horizon income (PV): ', round(income_project,2), ' €')
    print('P2G(H2) income(%): ', round(h2_income_project/income_project*100,2),' %')
    print('Salvage income(%): ', round(salv_inc_project/income_project*100,2),' %')
    print('Company horizon net profit (nominal): ', round(Net_nom_profit_project,2),' €' )
    print('Return on Investment (ROI) (%): ',  round(Net_nom_profit_project/capex_costs_project*100,2), ' %')
    print('Investment NPV: ', round(NPV,2) ,' €')


    print('==============================================')
    print('Power generation by wind energy(% of demand):', round(energy_from_wind_generation/total_el_energy_production*100,2),'%')
    print('Power generation by solar energy(% of demand):', round(energy_from_solar_generation/total_el_energy_production*100,2),'%')
    print('Average electricity prouction cost: ', average_el_price_per_kWh, ' €/kWh')

    print('\n =======================================')
    print('ECONOMIC STATISTICS')
    print('Investment NPV (should be zero): ',round(NPV,2) )
    print('H2 sale price : ', round(H2_sale_price_per_kg,2), '€/kg')
    print('LCOH: ', round(LCOH,5), '€/kg')

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
    print("GHG yearly emissions of baseline:", GHG_total_emissions_baseline_yearly, ' tons CO2 eq.')
    print("GHG emissions of scenario (% of baseline): ", round(GHG_emissions_fraction_of_baseline*100,2), ' %')
    print("GHG yearly emissions savings: ", round(GHG_total_emissions_baseline_yearly - GHG_total_emissions_scenario_yearly,3), ' tons CO2 eq. \n')
    
    #%%################## WRITE RESULTS TO CSV #############################
    #df = pd.DataFrame()
    data = {'description': recognition_string,
            'Sensitivity analysis scenario': sensitivity_analysis_scenario,
            'CAPEX(EUR@PV)': [round(capex_costs_project,2)],'Wind capex (%)': [round(capex_WF_project/capex_costs_project*100,2)],'Solar capex (%)': [round(capex_SF_project/capex_costs_project*100,2)],'Electrolysis capex (%)': [round(capex_electrolyzer_project/capex_costs_project*100,2)],'H2 storage capex (%)': [round(capex_H2_storage_project/capex_costs_project*100,2)],
            'OPEX(EUR@PV)': round(opex_costs_project,2) , 'Wind Fix.Opex(%)':round(fixed_opex_WF_project/opex_costs_project*100,2) ,'Wind Var.Opex(%)': round(wind_var_opex_project/opex_costs_project*100,2),
            'Solar Fix.Opex(%)': round(fixed_opex_SF_project/opex_costs_project*100,2),'Solar Var.Opex(%)': round(solar_var_opex_project/opex_costs_project*100,2),
            'Electrolysis Fix.Opex(%)': round(fixed_opex_electrolysis_project/opex_costs_project*100,2), 'Electrolysis Var.Opex(%)': round(electrolysis_var_opex_project/opex_costs_project*100,2),
            'H2 storage Fix.Opex(%)': round(fixed_opex_H2_storage_project/opex_costs_project*100,2) , 'H2 storage Var.Opex(%)': round(H2_storage_var_opex_costs_sim/opex_costs_project*100,2) ,
            'Replacement costs (EUR, PV)': round(rep_costs_project,2), 'Electrolysis repl. (%)': round(rep_costs_electrolyzer_project/rep_costs_project*100,2), 'H2 storage repl. (%)': round(rep_costs_storage_project/rep_costs_project*100,2),
            'Obj.val (-ann.cash flow EUR)': round(model_objval,2), 'Theoretical obj.val (_NPV EUR)': round(theoretical_objval,2), 'Difference (%) ': round((model_objval - theoretical_objval)/model_objval*100,4),
            'Company horizon costs(EUR@PV)': round(costs_project,2), 'Capex(%)': round(capex_costs_project/costs_project*100,2),'Opex(%)':round(opex_costs_project/costs_project*100,2) , 'Replacement(%)': round(rep_costs_project/costs_project*100,2),
            'Company horizon income(EUR@PV)':round(income_project,2) , 'P2G(H2) income(%)':  round(h2_income_project/income_project*100,2),'Salvage income(%)': round(salv_inc_project/income_project*100,2),'Company horizon net nom.profit(EUR)':round(Net_nom_profit_project,2) ,
            'ROI(%)': round(Net_nom_profit_project/capex_costs_project*100,2),
            'Wind generation(%)': round(energy_from_wind_generation/total_el_energy_production*100,2), 'Solar generation(%)': round(energy_from_solar_generation/total_el_energy_production*100,2),
            'Average electricity prod.cost(EUR/kWh)': average_el_price_per_kWh,'Investment NPV (should be zero)': round(NPV,2), 'H2 sale price(EUR/kg)': round(H2_sale_price_per_kg,2),  'LCOH(EUR/kg)': round(LCOH,5),
            'Wind nom.installation(MW)': round(network.generators.loc['wind_provider_PPA','p_nom_opt'],5), 'Wind av.capacity factor(% of p_nom)': round(wind_av_LF*100,2),
            'Solar nom.installation(MW)': round(network.generators.loc['solar_provider_PPA','p_nom_opt'],5), 'Solar av.capacity factor(% of p_nom)': round(solar_av_LF*100,2),
            'Electrolysis nominal installation(MW)': round(network.links.T.P_to_H2['p_nom_opt'],4) , 'Electrolysis av.capacity factor(% of p_nom)': round(electrolysis_av_LF*100,5),
            'H2 storage size (kg)': H2_storage_capacity_kg ,'H2 storage av.storage level (%)': round(H2_storage_av_level*100,2), 'H2 av.injection per volume (%)': round(H2_average_injection_ratio_per_volume*100,4) ,
            'H2 total project production (tons, PV):': round(H2_production_project,3), 'H2 ann. yearly production (tons):': round(H2_production_ann,3),
            'NG energy demand covered by synthetic H2 (%)':round(H2_content_of_NG*100,2) ,
            'GHG emissions of baseline yearly(tons CO2 eq.)':GHG_total_emissions_baseline_yearly ,'GHG emissions of scenario (% of baseline)': round(GHG_emissions_fraction_of_baseline*100,2),'GHG emissions savings (tons CO2 eq.)': round(GHG_total_emissions_baseline - GHG_total_emissions_scenario,2),
            'Duration of experiment (h)': experiment_duration
            }

    # Save the results to csv
    df = pd.DataFrame(data)
    df = df.T
    #H2_sale_price_per_kg,H2_selling_price_per_kg =3.15, 3.15
    save_results_dir =  f'./ATH_S1_{simulation_years}Y_{sensitivity_analysis_scenario}_H2_price_{H2_sale_price_per_kg}_EUR_per_kg'
    save_results_path = f'./Results/{sensitivity_analysis_scenario}' 
    if not os.path.exists(save_results_path):
         os.makedirs(save_results_path)
    df.to_csv(save_results_path + save_results_dir)


    print(f'===========END OF EXPERIMENT WITH H2 SALE VALUE {H2_sale_price_per_kg}. ===================')
    
    #%%################### WRITE USEFUL TIMESERIES RESULTS TO CSV ####################
    wind_nom_capacity, solar_nom_capacity = network.generators.loc['wind_provider_PPA','p_nom_opt'], network.generators.loc['solar_provider_PPA','p_nom_opt']
    actual_wind_cf_ts, actual_solar_cf_ts = network.generators_t.p['wind_provider_PPA'] / wind_nom_capacity , network.generators_t.p['solar_provider_PPA'] / solar_nom_capacity
    ng_supply_ts = network.generators_t.p['NG Generator']

    electrolysis_capacity = round(network.links.T.P_to_H2['p_nom_opt'],4)
    electrolysis_cf_ts = round(network.links_t.p0['P_to_H2']/ electrolysis_capacity ,4)

    H2_energy_storage_capacity = network.stores.loc['H2 depot', 'e_nom_opt' ]
    H2_energy_storage_cf_ts = round(network.stores_t.e['H2 depot'] / H2_energy_storage_capacity,3)

    H2_storage_charges_ts = network.stores_t.p['H2 depot']
    H2_injection_to_grid_ts = -network.links_t.p1['H2_to_NG'] #in MWh thermal. Divide with HHV_H2 to obtain kg of H2

    horizontal_concat = pd.concat([actual_wind_cf_ts, actual_solar_cf_ts,ng_supply_ts, electrolysis_cf_ts,H2_energy_storage_cf_ts,H2_storage_charges_ts,H2_injection_to_grid_ts], axis=1)    
    save_results_dir =  f'./timeseries_results_S1_{simulation_years}Y_{sensitivity_analysis_scenario}_p{H2_sale_price_per_kg}'
    horizontal_concat.to_csv(save_results_path + save_results_dir)


#%%Main function of the model. Uses argparse to put the "experiment function" into multiprocessing
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multiprocessing with argparse, with multiple H2 sale prices as parameters.")
    parser.add_argument('-hp1',"--hydrogen_price1", type=float, default= 3.5, help="H2 price value ")
    parser.add_argument('-hp2',"--hydrogen_price2", type=float, default= 4, help="H2 price value ")
    parser.add_argument('-hp3',"--hydrogen_price3", type=float, default= 5, help="H2 price value ")
    parser.add_argument('-hp4',"--hydrogen_price4", type=float, default= 6, help="H2 price value ")
    parser.add_argument('-hp5',"--hydrogen_price5", type=float, default= 8, help="H2 price value ")
    parser.add_argument('-y',"--simulation_years", type=int, default= 1, help="Number of simulation horizon years (integer). From 1 to 5 ")
    args = parser.parse_args()
    H2_sales_prices_list = [args.hydrogen_price1, args.hydrogen_price2, args.hydrogen_price3,args.hydrogen_price4, args.hydrogen_price5]
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

