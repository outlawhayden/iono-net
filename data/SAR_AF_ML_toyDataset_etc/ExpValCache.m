classdef ExpValCache < handle
    properties (Access = private)
        cacheArr
        storedPsi % for ML dataset only
    end   
    
    methods 
        function obj = ExpValCache() 
        end
        
        function ret = psiExpFun(obj, x_idx, yz_idx, isPlus_i, xi)
            % disable for acceleration 
            %psiFunFourier.checkVectorIndices(x_idx, yz_idx); 
            assert(xi == 0.5); 
                        
            cacheIdx = x_idx + yz_idx - 1; 
            expVal = obj.cacheArr(cacheIdx); 
            
            if isPlus_i
                ret = expVal; 
            else
                ret = conj(expVal); 
            end
        end
        
        function initializeCache(obj, psiFunObj, setup)
            assert(setup.xi == 0.5); 
            
            cacheArgStep = setup.steps.usc; 
            assert(cacheArgStep == setup.steps.I);
            assert(cacheArgStep == setup.steps.nu); 

            domainLims = BareboneUtilsIonoAF.getSpeckleDomainLims(setup); 
            assert(rem(domainLims(1),   cacheArgStep) == 0); 
            assert(rem(domainLims(end), cacheArgStep) == 0); 
            
            cacheArg = domainLims(1) : cacheArgStep/2 : domainLims(2); 
            assert(cacheArg(end) == domainLims(2)); 
            
            cachePsiVal = psiFunObj.psiValFun(cacheArg, setup); 
            obj.cacheArr = exp(1i * cachePsiVal); 
            
            % for ML dataset only
            obj.storedPsi.arg = cacheArg; 
            obj.storedPsi.val = cachePsiVal; 
        end
        
        % for ML dataset only
        function storedPsi = getStoredPsi(obj)
            storedPsi = obj.storedPsi; 
        end
    end      
    
end