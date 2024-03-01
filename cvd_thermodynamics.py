# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:46:02 2016
Modified october 2019: Create a list of calculated parameters, for graphs

Program to emulate CVD thermodinamics
Based on this work
http://kitchingroup.cheme.cmu.edu/blog/2013/03/01/Finding-equilibrium-composition-by-direct-minimization-of-Gibbs-free-energy-on-mole-numbers/
http://kitchingroup.cheme.cmu.edu/blog/2013/03/01/Gibbs-energy-minimization-and-the-NIST-webbook/
http://matlab.cheme.cmu.edu/2011/12/12/water-gas-shift-equilibria-via-the-nist-webbook/
http://kitchingroup.cheme.cmu.edu/blog/category/optimization/2/
http://matlab.cheme.cmu.edu/2011/12/20/the-gibbs-free-energy-of-a-reacting-mixture-and-the-equilibrium-composition/
Values from CRC Handbook of Chemistry and Physics 
p. 847 codata H0 S0
p. 851 G0

Equations (Shomate):
http://kitchingroup.cheme.cmu.edu/blog/2013/02/01/Water-gas-shift-equilibria-via-the-NIST-Webbook/
Cp° = A + B*t + C*t2 + D*t3 + E/t2
H° - H°298.15= A*t + B*t2/2 + C*t3/3 + D*t4/4 - E/t + F - H
S° = A*ln(t) + B*t + C*t2/2 + D*t3/3 - E/(2*t2) + G
G° = H°298.15 + H° - T * S°
Gtotal/(RT) = G°j / R / T + np.log((nj / Enj)*(P/Pstandard)))

@author: Alberto Bosca
All rights reserved
"""
import sympy as sy # Symbol,simplify,limit(func,var,val),diff(func,var),integrate(6*x**5, x),evalf,subs
import numpy as np
from scipy.constants import R, bar#, N_A, k, physical_constants,zero_Celsius
from scipy.optimize import fmin_slsqp #from the web

def show_formulas():
    """Calculate Shomate formulas"""
    A_sho, B_sho, C_sho, D_sho, E_sho, F_sho, G_sho, H_sho = sy.symbols('A B C D E F G H')
    T = sy.Symbol("T")
    temperature = 1000.0 # K
    H_298 = sy.Symbol("H_298")
    t = T/1000
    H_0 = H_298 + (A_sho*t + B_sho*t**2/2 + C_sho*t**3/3 + D_sho*t**4/4 - E_sho/t + F_sho - H_sho) #J/mol
    S_0 = A_sho*sy.ln(t) + B_sho*t + C_sho*t**2/2 + D_sho*t**3/3 - E_sho/(2*t**2) + G_sho
    C_shop_0 = A_sho + B_sho*t + C_sho*t**2 + D_sho*t**3 + E_sho/t**2
    G_0 = (H_0 - T*S_0)
    G_02 = (-T*sy.integrate(H_0/T**2,T))
    res1= G_0.subs({T:temperature, A_sho:40.68697, B_sho:40.73279, C_sho:-16.17840,
                    D_sho:3.669741, E_sho: -0.658411, F_sho:210.7067, G_sho:235.0052,
                    H_sho:226.7314, H_298 :227.4*1000.0})
    res2= G_02.subs({T:temperature, A_sho:40.68697, B_sho:40.73279, C_sho:-16.17840,
                     D_sho:3.669741, E_sho: -0.658411, F_sho:210.7067, G_sho:235.0052,
                     H_sho:226.7314, H_298 :227.4*1000.0})
    res3= H_0.subs({T:temperature, A_sho:40.68697, B_sho:40.73279, C_sho:-16.17840,
                    D_sho:3.669741, E_sho: -0.658411, F_sho:210.7067, G_sho:235.0052,
                    H_sho:226.7314, H_298 :227.4*1000.0})
    res4= S_0.subs({T:temperature, A_sho:40.68697, B_sho:40.73279, C_sho:-16.17840,
                    D_sho:3.669741, E_sho: -0.658411, F_sho:210.7067, G_sho:235.0052,
                    H_sho:226.7314, H_298 :227.4*1000.0})
    res5= C_shop_0.subs({T:temperature, A_sho:40.68697, B_sho:40.73279, C_sho:-16.17840,
                         D_sho:3.669741, E_sho: -0.658411, F_sho:210.7067, G_sho:235.0052,
                         H_sho:226.7314, H_298 :227.4*1000.0})
    print("G0",res1)
    print("G0_2",res2)
    print("H0",res3)
    print("S0",res4)
    print("Cp0",res5)
#compunds with [Name,formation_enthalpy, formation gibbs, entropy, cp]

class Compound(object):
    """Data structure for all compunds
    (phase, thermo, elements,shomate coefficients)
    Need to import numpy as np for arrays
    Also sympy is needed
    """
    def __init__ (self, phase = "gas", thermo = ["X", 0, 0, 0, 0], 
                  elements = [], shomate = [0, 0, 0, 0, 0, 0, 0, 0]):
        """Initialize the object"""
        self.pstandard = 100000.0 #Pa
        self.phase = phase
        self.formula = thermo[0]
        self.enthalpy = thermo[1]
        self.gibbs = thermo[2]
        self.entropy = thermo[3]
        self.cp = thermo[4]
        self.dict_thermo = dict(zip(['H_298','G_298','S_298','Cp_298'], thermo[1:]))
        self.elements = elements
        self.shomate = np.array(shomate)
        self.dict_shomate = dict(zip(['A_sho','B_sho','C_sho',
                                      'D_sho','E_sho','F_sho',
                                      'G_sho','H_sho'], shomate))
        self.H_0 = 0
        self.S_0 = 0
        self.Cp_0 = 0
        self.G_0 = 0
        self.S_T = 0
        self.Cp_T = 0
        self.G_T = 0
        self.formulas()

    def formulas (self):
        """Create sympy formulas to get values"""
        A_sho, B_sho, C_sho, D_sho, E_sho, F_sho, G_sho, H_sho, T_sho = sy.symbols('A_sho B_sho C_sho D_sho E_sho F_sho G_sho H_sho T_sho')
        H_298, G_298, S_298, Cp_298 = sy.symbols('H_298 G_298 S_298 Cp_298')
        t = T_sho/1000.0
        self.H_0 = H_298 + (A_sho*t + B_sho*t**2/2.0 + C_sho*t**3/3.0 + D_sho*t**4/4.0 - E_sho/t + F_sho - H_sho)*1000.0 #J/mol        
        self.S_0 = A_sho*sy.ln(t) + B_sho*t + C_sho*t**2/2.0 + D_sho*t**3/3 - E_sho/(2.0*t**2) + G_sho
        self.Cp_0 = A_sho + B_sho*t + C_sho*t**2 + D_sho*t**3 + E_sho/t**2
        self.G_0 = (self.H_0 - T_sho*self.S_0) #Alternative formula not working
#        self.G_0 = (-T_sho*sy.integrate(self.H_0/T_sho**2,T_sho))
        self.H_T = self.H_0.subs(self.dict_shomate).subs(self.dict_thermo)
        self.S_T = self.S_0.subs(self.dict_shomate).subs(self.dict_thermo)
        self.Cp_T = self.Cp_0.subs(self.dict_shomate).subs(self.dict_thermo)
        self.G_T = self.G_0.subs(self.dict_shomate).subs(self.dict_thermo)
              
    def G_T_p_x (self, T = 298.15, p = 100000.0, x = 1.0 ):
        """Function to evaluate the gibbs energy at a certain temperature,
        pressure and molar fraction"""
        #print(p,x,p <= 0.0,x <= 0.0)
        if ( (p <= 0.0) or (x <= 0.0) ):
            return 0.0
        else:
            return (self.G_T.subs({'T_sho':T}) + R*T*sy.ln(x*p/self.pstandard)).evalf() # added +1
#            return (self.G_T.subs({'T_sho':T}) - R*T*sy.ln(x*p/self.pstandard)).evalf() # added +1

    def print_values (self):
        """Check stored values in console"""
        print("Phase: ",self.phase)
        print("Chemical formula: ",self.formula)
        print("Formation enthalpy: ",self.enthalpy)
        print("Formation Gibbs energy: ",self.gibbs)
        print("Entropy: ",self.entropy)
        print("Cp: ",self.cp)
        print("Elements:",self.elements)

class CompoundDatabase(object):
    def __init__ (self):
        shomate = [-0.703029, 108.4773, -42.52157,
                   5.862788, 0.678565, -76.84376, 158.7163, -74.87310]
        self.methane = Compound("gas",["CH4", -74.6*1000.0, -50.5*1000.0, 186.3, 35.7],
                                {"H":4,"C":1}, shomate) #-74.6 -50.5 186.3 35.7

        shomate = [28.13786	, 36.74736, -4.347218,
                   -1.595673, 0.001860, 135.7118, 217.4814	, 145.6873]
        self.methyl = Compound("gas",["CH3", 145.7*1000.0, 147.9*1000.0, 194.2, 38.7],
                               {"H":3,"C":1}, shomate) # 145.7 147.9 194.2 38.7

        shomate = [31.96823, 6.783603, 12.51890,
                   -5.696265, -0.031115, 376.3558, 229.9150, 386.3924]
        self.methylene = Compound("gas",["CH2", 390.4*1000.0, 372.9*1000.0, 194.9, 33.8],
                                  {"H":2,"C":1}, shomate)# 390.4 372.9 194.9 33.8

        shomate = [67.47244, 11.75110, -2.021470,
                   0.136195, -9.806418, 185.4550, 253.5337, 226.7314]
        self.acetylene = Compound("gas",["C2H2", 227.4*1000.0, 209.9*1000.0, 200.9, 44.0],
                                  {"H":2,"C":2}, shomate) # 227.4 209.9 200.9 44.0

        shomate = [0, 0, 0, 0, 0, 0, 0, 0]
        self.graphite_gas = Compound("gas",["C", 716.7*1000.0, 671.3*1000.0, 158.1, 20.8],
                                     {"C":1}, shomate) #716.7 671.3 158.1 20.8

        shomate = [0, 0, 0, 0, 0, 0, 0, 0]
        self.graphite_crystal = Compound("solid",["C", 0.0*1000.0, 0.0, 5.7, 8.5],
                                         {"C":1}, shomate) # 0.0 ? 5.7 8.5

        shomate = [0, 0, 0, 0, 0, 0, 0, 0]
        self.dicarbon_gas = Compound("gas",["C2", 831.9*1000.0, 775.9*1000.0, 199.4, 43.2],
                                     {"C":2}, shomate) #831.9 775.9 199.4 43.2

        shomate = [20.78600, 2.825911e-7, -1.464191e-7,
                   1.092131e-8, -3.661371e-8, -6.197350, 179.9990, 0.0]
        self.argon = Compound("gas",["Ar", 0.0*1000.0, 0.0, 154.8, 20.8],
                              {"Ar":1}, shomate) # 0.0 NaN 154.8 20.8

        shomate = [18.563083, 12.257357, -2.859786,
                   0.268238, 1.977990, -1.147438, 156.288133, 0.0]
        self.hydrogen = Compound("gas",["H2", 0.0*1000.0, 0.0, 130.7, 28.8],
                                 {"H":2}, shomate) # 0.0 NaN 130.7 28.8

        shomate = [20.78603, 4.850638e-10, -1.582916e-10,
                   1.525102e-11, 3.196347e-11, 211.8020, 139.8711, 217.9994]
        self.hydrogen_radical = Compound("gas",["H", 218.0*1000.0, 203.3*1000.0, 114.7, 20.8],
                                         {"H":1}, shomate) #218.0 203.3 114.7 20.8

def gibbs_minimization(T_reaction_C = 1030, P_reaction_mbar = 80.0, flow_income_list = [5.0, 1000.0, 6000.0], verbose = True):
    """Program to minimize the gibbs energy
    @param: T_reaction_C (temperature in Celsius)
    @param: P_reaction_mbar (pressure in millibar)
    @param: molar_income_list [CH4, H2, Ar total] (Volumetric Flows in SCCM)
    @return: flow outcome [CH4, H2, Ar total, C2H2, H_radical]"""
    db = CompoundDatabase() # Load the compound database
    mixture = [db.methane, db.hydrogen, db.argon,db.acetylene, db.hydrogen_radical]
    T_reaction = float(T_reaction_C) + 273.0 #K (0.15 quitado a propósito para no tomar valores frontera)
    P_reaction = float(P_reaction_mbar)*bar/1000.0 # *bar/1000 #convert from mbar to Pa
    #sccm_to_mol_per_s = 1.0e-6/60.0*physical_constants["standard-state pressure"][0]/(R*zero_Celsius)
    molar_income = np.array(flow_income_list + [0.0, 0.0])#*sccm_to_mol_per_s
    element_balance_matrix = np.array(([1, 0, 0, 2, 0],  #Carbon balance
                                      [4, 2, 0, 2, 1],  #Hydrogen balance
                                      [0, 0, 1, 0, 0]))  #Argon balance
    #Unit conversion from sccm to mol per second
    #sccm_to_mol_per_s = 1.0e-6/60.0*physical_constants["standard-state pressure"][0]/(R*zero_Celsius)

    def matter_conservation(molar_composition):
        """The moles of each element must be conserved"""
        return (np.dot(element_balance_matrix, np.array(molar_composition)) - 
                np.dot(element_balance_matrix,molar_income))

    def ineq_constrains(molar_composition):
        """All positive values"""
        return molar_composition

    def gibbs_sumatory_T_p_x (mixture = [], molar_inlet = np.array([]), T_reaction = 273.15, P_reaction = 10000.0):
        """Function that returns the Gibbs energy for a certain composition"""
        sumatory = 0.0
        total_mols = sum(molar_inlet)
    #    print(total_mols)
        for i, single_compound in enumerate(mixture):
    #        print(i,molar_inlet[i],total_mols)
    #        a = molar_inlet[i]*single_compound.G_T_p_x(T_reaction, P_reaction, molar_inlet[i]/total_mols)
    #        print(a)
            sumatory +=  molar_inlet[i]*single_compound.G_T_p_x(T_reaction, P_reaction, molar_inlet[i]/total_mols)
#        print(sumatory)
        return sumatory

    def gibbs_sumatory_x(molar_composition):
        return gibbs_sumatory_T_p_x (mixture, molar_composition,T_reaction,P_reaction)      

#    a = Gibbs_sumatory_T_p_x (mixture, molar_income,T_reaction,P_reaction)
#    b = Gibbs_sumatory_x (molar_income)
#    print(a,b)
    initial_guess = flow_income_list  + [0, 0]
    result_optimization = fmin_slsqp(gibbs_sumatory_x, initial_guess, 
                                     f_eqcons = matter_conservation,
                                     f_ieqcons = ineq_constrains,
                                     iter=900, acc=1e-18, iprint = int(verbose))
    if (verbose):
        print('\nThermodinamic conditions:')
        print('{} {}'.format("Pressure(Pa)", P_reaction))
        print('{} {}'.format("Temperature(K)", T_reaction))
        print('\nInitial Flow:')
        print(np.dot(element_balance_matrix,molar_income))
        for s,x in zip(mixture, molar_income):
            print('{} {}'.format(s.formula, x))
        print('\nFinal conditions:')
        for s,x in zip(mixture, result_optimization):
            print('{} {}'.format(s.formula, x))

        print('\nRatios:')
        print('{} {}'.format("C2H2/CH4", result_optimization[3]/result_optimization[0]))
        print('{} {}'.format("H2/CH4", result_optimization[1]/result_optimization[0]))
        print('{} {}'.format("H/H2", result_optimization[4]/result_optimization[1]))
        print('{} {}'.format("C2H2/H", result_optimization[3]/result_optimization[4]))
        # check that constraints were met
        print(np.dot(element_balance_matrix, result_optimization) - 
                    np.dot(element_balance_matrix,molar_income))
        print (np.all(np.abs(np.dot(element_balance_matrix, result_optimization) 
        - np.dot(element_balance_matrix,molar_income))) < 1e-12)  
    return result_optimization
#    return (np.array(result_optimization)/sccm_to_mol_per_s).tolist()#

def thermodynamics_map_P(P_range, T_reaction,flow_income,number_of_points):
    """Small script for saving the thermodynamical parameter space.
    Changes pressure values.
    Call example:
       thermodynamics_map_P([50,250],990,[2,800,1800],10) """

    #Create list of values
    temporarylist =[] #Here save all values (input and output for each parameter variation)
    P_list = np.linspace(P_range[0],P_range[1],number_of_points)   

    for p in P_list:
        results_thermo = gibbs_minimization(T_reaction, p, flow_income,False)
        counter_recalulate = 0 #This is needed to avoid negative values (bad results)
        while((results_thermo < 0.0).any() or (counter_recalulate > 10)):
            counter_recalulate += 1
            print(f"Attemp no {counter_recalulate}")
            multiplier_shake = (10000.1001 + counter_recalulate)/10000.202
            #just move a bit the temperature parameter, less than a degree
            results_thermo = gibbs_minimization(T_reaction*multiplier_shake, p, flow_income,False)

        temporarylist.append([T_reaction, p,
                              flow_income[0], flow_income[1], flow_income[2],
                              results_thermo[0], results_thermo[1],
                              results_thermo[3], results_thermo[4],
                              results_thermo[3]/results_thermo[0],
                              results_thermo[1]/results_thermo[0],
                              results_thermo[4]/results_thermo[1],
                              results_thermo[3]/results_thermo[4],
                              100.*results_thermo[3]/flow_income[0],
                              100.*results_thermo[4]/flow_income[1]])

    #----Create readable header
    labelsparameters = ["T_growth_(C)", "P _(mbar)", 
                        "CH4_in_(sccm)", "H2_in_(sccm)", "Ar_in_(sccm)",
                        "Thermo_calc_CH4_(sccm)", "Thermo_calc_H2_(sccm)", 
                        "Thermo_calc_C2H2_(sccm)","Thermo_calc_H_rad_(sccm)",
                        "C2H2/CH4", 
                        "H2/CH4",
                        "H/H2", 
                        "C2H2/H",
                        "Methane_decomposition(%)",
                        "Hydrogen_decomposition(%)"]      
    temporaryheader = ""
    for singleheader in labelsparameters:
        temporaryheader+= singleheader + ";"
    #----End header
    #Save all the values (initial, calculated)
    np.savetxt("P_modification.csv",temporarylist,header= temporaryheader, fmt= "%s", delimiter=";",comments='')

def thermodynamics_map_T(P_reaction, T_range, flow_income, number_of_points):
    """Small script for saving the thermodynamical parameter space.
    Changes temperature values.
    Call example:
       thermodynamics_map_T(50,[910,1050],[2,800,1800],100) """

    #Create list of values
    temporary_list = [] #Here save all values (input and output for each parameter variation)
    T_list = np.linspace(T_range[0], T_range[1], number_of_points)

    for T_reaction in T_list:
        results_thermo = gibbs_minimization(T_reaction, P_reaction, flow_income, False)
        counter_recalulate = 0 #This is needed to avoid negative values (bad results)
        while((results_thermo < 0.0).any() or (counter_recalulate > 10)):
            counter_recalulate += 1
            print(f"Attemp no {counter_recalulate}")
            multiplier_shake = (10000.1001 + counter_recalulate)/10000.202
            #just move a bit the temperature parameter, less than a degree
            results_thermo = gibbs_minimization(T_reaction*multiplier_shake, P_reaction, flow_income, False)

        temporary_list.append([T_reaction, P_reaction,
                              flow_income[0], flow_income[1], flow_income[2],
                              results_thermo[0], results_thermo[1],
                              results_thermo[3], results_thermo[4],
                              results_thermo[3]/results_thermo[0],
                              results_thermo[1]/results_thermo[0],
                              results_thermo[4]/results_thermo[1],
                              results_thermo[3]/results_thermo[4],
                              100.*results_thermo[3]/flow_income[0],
                              100.*results_thermo[4]/flow_income[1]])

    #----Create readable header
    labels_parameters = ["T_growth_(C)", "P _(mbar)",
                        "CH4_in_(sccm)", "H2_in_(sccm)", "Ar_in_(sccm)",
                        "Thermo_calc_CH4_(sccm)", "Thermo_calc_H2_(sccm)", 
                        "Thermo_calc_C2H2_(sccm)","Thermo_calc_H_rad_(sccm)",
                        "C2H2/CH4", 
                        "H2/CH4",
                        "H/H2", 
                        "C2H2/H",
                        "Methane_decomposition(%)",
                        "Hydrogen_decomposition(%)"]      
    temporary_header = ""
    for single_header in labels_parameters:
        temporary_header+= single_header + ";"
    #----End header
    #Save all the values (initial, calculated)
    np.savetxt("T_modification.csv",temporary_list, header= temporary_header, fmt= "%s", delimiter=";", comments='')

def thermodynamics_map_H2(P_reaction, T_reaction, flow_income_range, number_of_points):
    """Small script for saving the thermodynamical parameter space.
    Changes hydrogen flux values.
    Call example:
       thermodynamics_map_H2(50,1050,[2,[80,800],1800],10) """

    #Create list of values
    temporarylist =[] #Here save all values (input and output for each parameter variation)
    H2_list = np.linspace(flow_income_range[1][0],flow_income_range[1][1],number_of_points)

    for H2_reaction in H2_list:
        flow_income = [flow_income_range[0],H2_reaction,flow_income_range[2]]
        results_thermo = gibbs_minimization(T_reaction, P_reaction, flow_income,False)
        counter_recalulate = 0 #This is needed to avoid negative values (bad results)
        while((results_thermo < 0.0).any() or (counter_recalulate > 10)):
            counter_recalulate += 1
            print(f"Attemp no {counter_recalulate}")
            multiplier_shake = (10000.1001 + counter_recalulate)/10000.202
            #just move a bit the temperature parameter, less than a degree
            results_thermo = gibbs_minimization(T_reaction*multiplier_shake, P_reaction, flow_income, False)
   
        temporarylist.append([T_reaction, P_reaction,
                              flow_income[0], flow_income[1], flow_income[2],
                              results_thermo[0], results_thermo[1],
                              results_thermo[3], results_thermo[4],
                              results_thermo[3]/results_thermo[0],
                              results_thermo[1]/results_thermo[0],
                              results_thermo[4]/results_thermo[1],
                              results_thermo[3]/results_thermo[4],
                              100.*results_thermo[3]/flow_income[0],
                              100.*results_thermo[4]/flow_income[1]])
    
    #----Create readable header
    labelsparameters = ["T_growth_(C)", "P _(mbar)",
                        "CH4_in_(sccm)", "H2_in_(sccm)", "Ar_in_(sccm)",
                        "Thermo_calc_CH4_(sccm)", "Thermo_calc_H2_(sccm)",
                        "Thermo_calc_C2H2_(sccm)","Thermo_calc_H_rad_(sccm)",
                        "C2H2/CH4",
                        "H2/CH4",
                        "H/H2",
                        "C2H2/H",
                        "Methane_decomposition(%)",
                        "Hydrogen_decomposition(%)"]    
    temporaryheader = ""
    for singleheader in labelsparameters:
        temporaryheader+= singleheader + ";"
    #----End header
    #Save all the values (initial, calculated)
    np.savetxt("H2_modification.csv", temporarylist, header= temporaryheader,
               fmt= "%s", delimiter=";", comments='')
    
if __name__ == '__main__':
#    show_formulas()
    thermodynamics_map_P([10,250],990,[2,800,1800],400)
    thermodynamics_map_T(50,[880,1090],[2,800,1800],400)
    thermodynamics_map_H2(50,1050,[2,[50,1000],1800],400)
#    res =gibbs_minimization()
