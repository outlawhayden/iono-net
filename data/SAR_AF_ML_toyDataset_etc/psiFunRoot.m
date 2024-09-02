classdef (Abstract) psiFunRoot < handle
    
    methods (Abstract)
        psiExpFun(obj, x_idx, yz_idx, isPlus_i, xi)
        psiValFun(obj, x, setup)
        psiInternals(obj)
    end   
  
end