classdef BareboneUtilsIonoAF < handle
    methods (Static)
        % used as setup.windowFun
        function ret = parabolicWindow(xDimless)
            rawWindow = 1 - (2*xDimless).^2; 
            ret = rawWindow / mean(rawWindow); 
        end
        
        % used as setup.windowFun
        function ret = parzenWindow(xDimless)
            %normConstant = 3/8; % makes the integral = 1, but the center istoo high for matched filtering 
            normConstant = 0.5; 
            
           x = xDimless * 2; 
            ret = zeros(size(x)); 
            
            maskInner = (abs(x) < 0.5); 
            ret(maskInner) = 1 - 6 * x(maskInner).^2 .* (1 - abs(x(maskInner))); 
            
            maskOuter = (abs(x) < 1) & (~maskInner); 
            ret(maskOuter) = 2 * (1 - abs(x(maskOuter))).^3; 
            
            ret = ret / normConstant; 
        end      
        
        % used as setup.sumFun
        function ret = sumSimpson(x, matrixDir)
            if nargin == 1
                % Simpson summation for a vector 
                assert(isvector(x)); 
                assert(rem(numel(x), 2) == 1); 
                ret = (1/3) * (x(1) + x(end) + ... 
                               4 * sum(x(2 : 2 : end - 1)) + ... 
                               2 * sum(x(3 : 2 : end - 2)));    
            else 
                % Simpson summation for a matrix, ROW-wise (dir = 2)
                assert((nargin == 2) && (~isvector(x)) && (matrixDir == 2)); 
                dir = 2; 
                dim2 = size(x, dir); assert(rem(dim2, 2) == 1); 
                ret = (1/3) * (x(:, 1) + x(:, end) + ... 
                               4 * sum(x(:, 2 : 2 : end - 1), dir) + ... 
                               2 * sum(x(:, 3 : 2 : end - 2), dir));                
            end
        end
        
        % used as setup.sumFun
        function ret = sumTrapz(x, matrixDir)
            if nargin == 1
                % Trapezoid summation for a vector 
                assert(isvector(x)); 
                assert(rem(numel(x), 2) == 1); 
                ret = (1/2) * (x(1) + x(end)) + ... 
                           sum(x(2 : end - 1));    
            else 
                % Trapezoid summation for a matrix, ROW-wise (dir = 2)
                assert((nargin == 2) && (~isvector(x)) && (matrixDir == 2)); 
                dir = 2; 
                dim2 = size(x, dir); assert(rem(dim2, 2) == 1); 
                ret = (1/2) * (x(:, 1) + x(:, end)) + ... 
                           sum(x(:, 2 : end - 1), dir);                             
            end
        end        
        
        %{
        function ret = rectangularWindow(xDimless)
            ret = ones(size(x)); 
        end
        %}
                
        function uscStruct = scene2signal_window(nuStruct, psiParams, setup)
            plus_i_inExp = true; 
            z = nuStruct.zarg; 
            x = min(z) : setup.steps.usc : max(z); 

            vals = nan(size(x)) + 1i * nan(size(x)); 
            for x_idx = 1:numel(x)
                xval = x(x_idx); 
                zlimForX = [xval - setup.F/2, xval + setup.F/2];            % THIS CREATES A RECTANGUALR WINDOW
                if (zlimForX(1) >= min(z)) && (zlimForX(2) <= max(z))       % THIS CREATES A RECTANGUALR WINDOW
                    zIntrvlMask = (z >= zlimForX(1)) & (z <= zlimForX(2));  % THIS CREATES A RECTANGUALR WINDOW
                    zIntrvl = z(zIntrvlMask); 
                    nuIntrvl = nuStruct.complVal(zIntrvlMask);            

                    firstExp = exp(1i * pi * (xval - zIntrvl).^2 / setup.F); 
                    
                    %psiArg = (1 - setup.xi) * zIntrvl + setup.xi * xval; 
                    %secondExp = exp(-1i * psiParams.psiFun(psiArg));
                    secondExp = psiParams.psiExpFun(x_idx, find(zIntrvlMask), ~plus_i_inExp, setup.xi); %#ok<FNDSB>
                    
                    % !!! NOTE: NO WINDOWING FOR SCENE -> SIGNAL except for the antenna aperture. 
                    %           There IS still aperture limitation because "usc" is a _scattered_ signal, 
                    %           with the incident signal affected by the aperture function of the antenna.   
                    %windowVal = setup.windowFun((xval - zIntrvl)/setup.F);               
                    %vals(x_idx) = setup.steps.nu * setup.sumFun(firstExp .* secondExp .* nuIntrvl .* windowVal); % Riemann sum
                    vals(x_idx) = setup.steps.nu * setup.sumFun(firstExp .* secondExp .* nuIntrvl); % Riemann sum
                end
            end

            uscStruct.xarg = x; 
            uscStruct.vals = vals; 
        end
        
        function [I, deI_de_pk, deI_de_qk] = signal2img_window(uscStruct, psiParams, setup, doUsePsi)
            plus_i_inExp = true; 
            provideGrad = (nargout > 1); 
            [x, y] = BareboneUtilsIonoAF.xyFrom_uscStruct(uscStruct, setup); 
            xStep = setup.steps.usc;  
            assert(xStep <= 1, 'Using xStep > 1 results in high sidelobes (found: %g)', xStep); 
                
            minindValidUsc = find(~isnan(uscStruct.vals), 1, 'first'); 
            maxindValidUsc = find(~isnan(uscStruct.vals), 1, 'last'); 
                
            vals = nan(size(y)); 
            if provideGrad
                assert(isrow(vals)); 
                vals_de_pk = nan(setup.ionoNharm, numel(vals)); 
                vals_de_qk = nan(setup.ionoNharm, numel(vals)); 
            end
            
            needKernelPart = true; 
            for y_idx = 1:numel(y)
                yval = y(y_idx); 
                xlimForY = [yval - setup.F/2, yval + setup.F/2]; 

                % usc is nan for the values of x near the ends of the synthetic aperture
                isUscForThisY_allValid = (xlimForY(1) >= uscStruct.xarg(minindValidUsc)) && ...  
                                         (xlimForY(2) <= uscStruct.xarg(maxindValidUsc)); 
                if isUscForThisY_allValid
                    xIntrvlMask = (x >= xlimForY(1)) & (x <= xlimForY(2)); 

                    xIntrvl = x(xIntrvlMask); 
                    uscIntrvl = uscStruct.vals(xIntrvlMask);            

                    if needKernelPart % computation of the part that depends on (y-x) will be called only once
                        firstExp = exp(-1i * pi * (yval - xIntrvl).^2 / setup.F); 
                        windowVal = setup.windowFun((yval - xIntrvl) / setup.F);   
                        kernelPart = firstExp .* windowVal; 
                        needKernelPart = false; 
                    end
                    psiArg = (1 - setup.xi) * yval + setup.xi * xIntrvl; 
                    if doUsePsi
                        %secondExp = exp(1i * psiParams.psiFun(psiArg));  
                        x_idx = find(xIntrvlMask); 
                        secondExp = psiParams.psiExpFun(x_idx, y_idx, plus_i_inExp, setup.xi);  %#ok<FNDSB>
                    else
                        secondExp = ones(size(psiArg)); 
                    end
                    
                    Iintegrand = kernelPart .* secondExp .* uscIntrvl; 
                    vals(y_idx) = xStep * setup.sumFun(Iintegrand); % Riemann sum
                    
                    if provideGrad
                        kPsi = BareboneUtilsIonoAF.get_kPsi_relToF(setup);    
                        assert(iscolumn(kPsi)); 
                        assert(isrow(psiArg)); 
                        trigArgMtx = kPsi * psiArg; 
                        
                        IintegrandMtx = repmat(Iintegrand, [setup.ionoNharm, 1]); 
                        % see p. "Gamma"3
                        termFor_pk =  1i * cos(trigArgMtx); 
                        termFor_qk = -1i * sin(trigArgMtx); 
                        vals_de_pk(:, y_idx) = xStep * setup.sumFun(termFor_pk .* IintegrandMtx, 2); 
                        vals_de_qk(:, y_idx) = xStep * setup.sumFun(termFor_qk .* IintegrandMtx, 2);                                       
                    end
                end
            end
            I.yarg     = y; 
            I.complVal = vals / setup.F; % normalize F*sinc(pi x) to work like \delta(x) 
            
            if provideGrad
                deI_de_pk = vals_de_pk / setup.F; % in agreement with the above normalization
                deI_de_qk = vals_de_qk / setup.F;
            end
        end
    
        function [x, y] = xyFrom_uscStruct(uscStruct, setup)
            x = uscStruct.xarg; 
            y = min(x) : setup.steps.I : max(x); 
        end
        
        function absval_smooth = getabsval_smooth(imgVal, setup)
            absval_smooth = sqrt(setup.smooth_l1_deltasq + conj(imgVal) .* imgVal); 
            assert(isreal(absval_smooth)); 
        end
        
        % see p. "Gamma"4
        function l1_smooth_grad = getl1_smooth_grad(imgWithPsiRaw, derivs, setup)
            imgval_ref = imgWithPsiRaw.complVal; 
            absval_smooth_ref = BareboneUtilsIonoAF.getabsval_smooth(imgval_ref, setup);   

            mask_nonan = ~isnan(imgval_ref); 
            conjI_over_L = conj(imgval_ref) ./ absval_smooth_ref; 
            conjI_over_L_Mat_nonan = repmat(conjI_over_L(mask_nonan), [setup.ionoNharm, 1]); 
            % the summation below is over pixels - should we extend Simpson summation to this as well? 
            l1_smooth_grad.depk = sum(real(conjI_over_L_Mat_nonan .* derivs.deI_de_pk(:, mask_nonan)), 2);     
            l1_smooth_grad.deqk = sum(real(conjI_over_L_Mat_nonan .* derivs.deI_de_qk(:, mask_nonan)), 2);  
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%
        function PSstruct = createPSstruct(psCoord, psAmpl, setup)
            % create delta peaks instead of narrow peaks   
            PSstruct.locs = psCoord; 
            PSstruct.ampl = psAmpl / setup.steps.nu; 
        end        
       
        function zarg = create_zarg(setup)
            zStep = setup.steps.nu; 
            domainLims = BareboneUtilsIonoAF.getSpeckleDomainLims(setup); 
            zarg = domainLims(1) : zStep : domainLims(2);
        end        
  
        function domainLims = getSpeckleDomainLims(setup)
            domainLims = [0, setup.domainLengthInF * setup.F]; 
        end
        
        function nuStruct = PSstruct2nuStruct_noAssert(PSstruct, zarg)
            % do NOT create nuStruct.avals
            nuStruct.zarg = zarg; 
            nuStruct.complVal = zeros(size(zarg)); 

            for ipeak = 1:numel(PSstruct.locs)
                loc = PSstruct.locs(ipeak);

                idx_zarg = find(zarg == loc);
                assert(numel(idx_zarg) == 1); 
                peakVal = PSstruct.ampl(ipeak);
                nuStruct.complVal(idx_zarg) = nuStruct.complVal(idx_zarg) + peakVal; % corrected; it is wrong without the first term if there are two PSs with the same coords
            end
        end        
        
        function nuStruct = PSstruct2nuStruct_assertMatch(PSstruct, zarg)
            % do NOT create nuStruct.avals
            nuStruct.zarg = zarg; 
            nuStruct.complVal = zeros(size(zarg)); 

            for ipeak = 1:numel(PSstruct.locs)
                loc = PSstruct.locs(ipeak);

                idx_zarg = find(zarg == loc);
                assert(numel(idx_zarg) == 1); 
                peakVal = PSstruct.ampl(ipeak);
                nuStruct.complVal(idx_zarg) = nuStruct.complVal(idx_zarg) + peakVal; % corrected; it is wrong without the first term if there are two PSs with the same coords
            end
        end
        
        function kPsi = get_kPsi_relToF(setup)
            kMin = (2*pi / setup.F) * setup.F_to_lmax; 
            kPsi = kMin * (1 : setup.ionoNharm)'; 
        end        
        
        function [imgStruct, uscStruct] = simulateSignalAndImage(psiParams, nuStruct, setup) 
            doCompensate = true;     

            % simulate the signal, compensated and uncompensated images
            uscStruct             = BareboneUtilsIonoAF.scene2signal_window(nuStruct, psiParams, setup);                %%% formula (B) %%%%%%%%%%%
            imgStruct.withPsi_raw = BareboneUtilsIonoAF.signal2img_window(uscStruct, psiParams, setup, doCompensate);   %%% formula (D) with compensation %%%%%%%
            imgStruct.wo_Psi_raw  = BareboneUtilsIonoAF.signal2img_window(uscStruct, psiParams, setup, ~doCompensate);  %%% formula (D) without compensation %%%%%%
        end        
        
        function checkPsiFunOutput(ret)
            assert(~any(isnan(ret(:)))); 
            assert(all(isreal(ret(:))));     
        end
        
        function compl_ampls = xDisassemblePsi(x, ionoNharm)
            %assert(numel(x) == 2 * ionoNharm); 
            % DEBUG
            if (numel(x) ~= 2 * ionoNharm)
                error('FIX THIS!!!')
                a = 1; 
            end
            compl_ampls = x(1 : ionoNharm) + 1i * x((ionoNharm + 1) : end); 
        end

        function x = xAssemblePsi(compl_ampls)
            x = [real(compl_ampls); ...  
                 imag(compl_ampls)]; 
        end                
    end
end