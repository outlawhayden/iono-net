classdef psiFunFourierWithInterpolant < psiFunFourier
    properties (Access = private)
        psiInterpolant
        xyz_arg_cache 
    end
    
    methods 
        function obj = psiFunFourierWithInterpolant(compl_ampls_in)
            obj@psiFunFourier(compl_ampls_in); 
        end
        
        function ret = psiExpFun(obj, x_idx, yz_idx, isPlus_i, xi)
            psiFunFourier.checkVectorIndices(x_idx, yz_idx); 
            
            x_val  = obj.xyz_arg_cache(x_idx); 
            yz_val = obj.xyz_arg_cache(yz_idx); 
            psiArg = (1 - xi) * yz_val + xi * x_val; 
            expVal = exp(1i * obj.psiFun(psiArg)); 
            if isPlus_i
                ret = expVal; 
            else
                ret = conj(expVal); 
            end
        end
        
        function createPsiInterpolant(obj, setup)
            domainLims = BareboneUtilsIonoAF.getSpeckleDomainLims(setup); 
            x = domainLims(1) : setup.ionoStep : domainLims(2); 
            val = obj.psiValFun(x, setup); 
            obj.psiInterpolant = griddedInterpolant(x, val, 'linear', 'none'); 
            
            obj.xyz_arg_cache = BareboneUtilsIonoAF.create_zarg(setup); 
        end
    end
    
    methods (Access = private)
        function ret = psiFun(obj, x)
            ret = obj.psiInterpolant(x); 
            BareboneUtilsIonoAF.checkPsiFunOutput(ret); 
        end        
    end
    
end
