classdef (Abstract) psiFunFourier < psiFunRoot
    properties (SetAccess = protected)
        compl_ampls
    end    
    
    methods
        function obj = psiFunFourier(compl_ampls_in)
            obj.compl_ampls = compl_ampls_in; 
        end
        
        function ret = psiInternals(obj)
            ret = obj.compl_ampls; 
        end        
        
        % called for initialization of cache in ExpValCache and for plotting
        function val = psiValFun(obj, x, setup) 
            assert(setup.ionoNharm == numel(obj.compl_ampls)); 
            
            kPsi = BareboneUtilsIonoAF.get_kPsi_relToF(setup); 
            val = zeros(size(x)); 
            for ik = 1 : setup.ionoNharm
                val = val + real(obj.compl_ampls(ik) * exp(1i * kPsi(ik) * x)); 
            end        
        end             
    end
    
 
    
    methods (Static, Access = protected)
        function checkVectorIndices(x_idx, yz_idx)
            assert(isvector(x_idx) || isvector(yz_idx)); 
        end
    end
  
end