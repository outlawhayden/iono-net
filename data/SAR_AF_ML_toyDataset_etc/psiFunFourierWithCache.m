classdef psiFunFourierWithCache < psiFunFourier
    properties (Access = private)
        expPsiValCache
    end
    
    methods 
        function obj = psiFunFourierWithCache(compl_ampls_in)
            obj@psiFunFourier(compl_ampls_in); 
            obj.expPsiValCache = ExpValCache(); 
        end
        
        %%%%% REFACTOR WITH psiFunPolynomial 
        function ret = psiExpFun(obj, x_idx, yz_idx, isPlus_i, xi)
            ret = obj.expPsiValCache.psiExpFun(x_idx, yz_idx, isPlus_i, xi); 
        end
        
        function initializeCache(obj, setup)
            obj.expPsiValCache.initializeCache(obj, setup); 
        end
        
        % for ML dataset only
        function storedPsi = getStoredPsi(obj)
            storedPsi = obj.expPsiValCache.getStoredPsi(); 
        end        
    end       
end
