function readAndBuildImage(iseed)
    if ~exist('iseed', 'var')
        iseed = 2; 
    end
    matFname = 'radarSeries.mat'; 
       
    plotImage_Signal_Screen(iseed, matFname)    
    plotScreenInMultipleWays(iseed, matFname)
    plotTwoNuStructs(iseed, matFname)    
end

function plotTwoNuStructs(iseed, matFname)
    S = load(matFname); 
    fprintf('Loading data from %s\n', matFname);   

    setup = S.dataset.meta.setup; 
    nuStructWithSpeckle = S.dataset.records{iseed}.nuStructs.withSpeckle;  
    PSstruct = S.dataset.records{iseed}.PSstruct; 
    nuStructWithoutSpeckle = S.dataset.records{iseed}.nuStructs.withoutSpeckle;     
    arg = S.dataset.meta.Z; 
    
    numPS = numel(PSstruct.locs); 
    
    figure('units', 'normalized', 'position', [0.1 0.1 0.5 0.3]);  
    assert(isequal(arg, nuStructWithSpeckle.zarg)); 
    assert(isequal(arg, nuStructWithoutSpeckle.zarg));     
    hold on; plot(arg, setup.steps.nu * abs(nuStructWithoutSpeckle.complVal), 'r-', 'linewidth', 6, 'displayname', 'without speckle')       
    hold on; plot(arg, setup.steps.nu * abs(nuStructWithSpeckle.complVal), 'k-', 'displayname', 'with speckle')      
   
    hl = legend; 
    set(hl, 'fontsize', 24, 'interpreter', 'latex')      
    xlim([min(arg), max(arg)])
    grid on
    title(sprintf('Test plot: reflectivity (Abs) * step, with and without speckle; point scatterers: %d', numPS))    
    
    jpgFname = sprintf('reflectivity_record%d.jpg', iseed); 
     print('-djpeg', jpgFname); 
end

function plotScreenInMultipleWays(iseed, matFname)
    S = load(matFname); 
    fprintf('Loading data from %s\n', matFname);   
 
    setup = S.dataset.meta.setup; 
    compl_ampls = S.dataset.records{iseed}.psiParams.compl_ampls; 
    storedPsi = S.dataset.records{iseed}.storedPsi; 
    storedPsi_dd_Val = S.dataset.records{iseed}.storedPsi_dd_Val; 
    
    psiParams = setup.createPsiImplFun(compl_ampls, setup); 
    kPsi = S.dataset.meta.kPsi;
       
    zarg = BareboneUtilsIonoAF.create_zarg(setup); 
    assert(isequal(S.dataset.meta.S, storedPsi.arg));     
    assert(isequal(S.dataset.meta.S(1:2:end), zarg)); % only for xi == 0.5

    figure('units', 'normalized', 'position', [0.1 0.1 0.5 0.4]);  
    psiValDirect = buildPsiVals(kPsi, compl_ampls, zarg);    
    psiValDirect_dd = diff(diff(psiValDirect)) / (setup.steps.nu)^2;     
    
    psiValInternal = psiParams.psiValFun(zarg, setup); 
    
    yyaxis left      
    hold on; plot(zarg, psiValDirect, 'm-', 'linewidth', 10, 'displayname', 'buildPsiVal')  
    hold on; plot(storedPsi.arg, storedPsi.val, 'k-', 'linewidth', 5, 'displayname', 'storedPsi')
    hold on; plot(zarg, psiValInternal, 'g-', 'linewidth', 2, 'displayname', 'psiValFun')     

    yyaxis right
    zarg_dd = zarg(2:end-1);     
    lightBlue = [0.3010 0.7450 0.9330]; 
    hold on; hlb = plot(zarg_dd, psiValDirect_dd, '-', 'linewidth', 8, 'displayname', 'diff(diff(psiValDirect))'); set(hlb, 'color', lightBlue); 
    hold on; plot(storedPsi.arg, storedPsi_dd_Val, 'k--', 'linewidth', 2, 'displayname', 'storedPsi_dd_Val')
    centerYlim()
    ylabel('\Psi''''(s)')
    xlabel('s')
     
    yyaxis left
    ylabel('\Psi(s)')
    yl = ylim; maxabs_yl = max(abs(yl));     
    ylim([-maxabs_yl, maxabs_yl]); 
    grid on; xlim([min(zarg), max(zarg)])        
 
    g = gca;     
    g.YAxis(1).Color = 'k'; 
    g.YAxis(2).Color = lightBlue;     
    hl = legend; set (hl, 'fontsize', 24, 'location', 'eastoutside', 'interpreter', 'none')
    
    title('Test plot: screen density and its second derivative')   
    
    jpgFname = sprintf('phaseScreen_record%d.jpg', iseed); 
     print('-djpeg', jpgFname);  
end

function centerYlim()
    yl = ylim; 
    maxabs_yl = max(abs(yl)); 
    ylim([-maxabs_yl, maxabs_yl])
end


function plotImage_Signal_Screen(iseed, matFname)
    plotNos.img = 2; 
    plotNos.screen = 3; 
    plotNos.total = 5; 

    doShowPeakHeight = true; 
    
    S = load(matFname); 
    fprintf('Loading data from %s\n', matFname);   
        
    setup = S.dataset.meta.setup; 
    uscStruct = S.dataset.records{iseed}.uscStruct; 
    nuStructWithSpeckle = S.dataset.records{iseed}.nuStructs.withSpeckle; 
    storedPsi = S.dataset.records{iseed}.storedPsi; 
    storedPsi_dd_Val = S.dataset.records{iseed}.storedPsi_dd_Val;     
    
    compl_ampls = S.dataset.records{iseed}.psiParams.compl_ampls; 
    psiParams = setup.createPsiImplFun(compl_ampls, setup); 
    PSstruct = S.dataset.records{iseed}.PSstruct; 
    kPsi = S.dataset.meta.kPsi;
               
    zarg = BareboneUtilsIonoAF.create_zarg(setup); 
    numPS = numel(PSstruct.locs); 
  
    figure('units', 'normalized', 'position', [0.1 0.1 0.5 0.8]);  
    
    subplot(plotNos.total, 1, 1)  
    assert(isequal(S.dataset.meta.Z, nuStructWithSpeckle.zarg)); 
    plot(S.dataset.meta.Z, setup.steps.nu * abs(nuStructWithSpeckle.complVal), 'k-', 'displayname', 'Abs')      
    hl = legend; 
    set(hl, 'fontsize', 24, 'interpreter', 'latex')      
    xlim([min(zarg), max(zarg)])
    grid on
    title(sprintf('Reflectivity (Abs) * step; point scatterers: %d', numPS))       
           
    displayTrueAndZeroImages(plotNos, uscStruct, kPsi, psiParams, zarg, storedPsi, storedPsi_dd_Val, setup);
    
    subplot(plotNos.total, 1, plotNos.img); 
    indicatePointScatterers( doShowPeakHeight, PSstruct, setup);    
    
    subplot(plotNos.total, 1, plotNos.screen); yyaxis left
    indicatePointScatterers(~doShowPeakHeight, PSstruct, setup);   
    
    subplot(plotNos.total, 1, 4)
    %mask = ~isnan(real(uscStruct.vals)); 
    assert(isequal(S.dataset.meta.X, uscStruct.xarg));
    hold on; plot(S.dataset.meta.X, real(uscStruct.vals), 'r-', 'linewidth', 2, 'displayname', 'Re')
    hold on; plot(S.dataset.meta.X, imag(uscStruct.vals), 'b-', 'linewidth', 2, 'displayname', 'Im')      
    hl = legend; 
    set(hl, 'fontsize', 24, 'interpreter', 'latex')      
    %ylabel('Re(u), Im(u)')
    yl = ylim; maxabs_yl = max(abs(yl));     
    ylim([-maxabs_yl, maxabs_yl]); 
    grid on; xlim([min(zarg), max(zarg)])         
    title('Antenna signal (Re,Im)')    
    
    subplot(plotNos.total, 1, 5)  
    hold on; plot(uscStruct.xarg, abs(uscStruct.vals), 'k-', 'linewidth', 2, 'displayname', 'Abs')      
    hl = legend; 
    set(hl, 'fontsize', 24, 'interpreter', 'latex')      
    %ylabel('abs(u)')
    grid on; xlim([min(zarg), max(zarg)])    
    title('Antenna signal (Abs)')
    
    xlabel('x, y, z, s')
    hsg = sgtitle(sprintf('Combined test plot: file ''%s'', record no. %d, window: %s, levels (clutter, noise) = (%g, %g)', ... 
                  matFname, iseed, setup.windowType, setup.addSpeckleCoeff, setup.relNoiseCoeff)); 
    set(hsg, 'fontsize', 28)
    
    jpgFname = sprintf('combinedPanels_record%d.jpg', iseed); 
     print('-djpeg', jpgFname);    
end

function indicatePointScatterers(isShowPeakHeight, PSstruct, setup)
    numPS = numel(PSstruct.locs); 
    for iPS = 1:numPS
        loc = PSstruct.locs(iPS); 
        if isShowPeakHeight
            absAmplScaled = abs(PSstruct.ampl(iPS)) * setup.steps.nu; 
        else
            absAmplScaled = 0; 
        end
        if iPS == 1
            hold on; plot([loc, loc], [0, absAmplScaled], 'rx-', 'linewidth', 2, 'displayname', 'point scatterers');
            if isShowPeakHeight % otherwise we don't want them in the legend - still kinda hack
                hl = legend; 
                set(hl, 'AutoUpdate', 'off')
            end
        else
            hold on; plot([loc, loc], [0, absAmplScaled], 'rx-', 'linewidth', 2); 
        end
    end   
    if isShowPeakHeight % otherwise we don't want them in the title too - still kinda hack    
       title('SAR images: with and without correction')  
    end
end

function displayTrueAndZeroImages(plotNos, uscStruct, kPsi, psiParams, zarg, storedPsi, storedPsi_dd_Val, setup)
    doCompensate = true; 
    
    subplot(plotNos.total, 1, plotNos.img)
    hl = findobj(gcf, 'Type', 'Legend');
    set(hl, 'AutoUpdate', 'on')            
    buildAndPlotImageCurve('b', 3, uscStruct, psiParams, setup, ~doCompensate, 'I[0]');  
    buildAndPlotImageCurve('m', 3, uscStruct, psiParams, setup, doCompensate, 'I[\Psi]');        
    set(hl, 'AutoUpdate', 'off')
    
    hl = legend; 
    set(hl, 'fontsize', 24)  
    %ylabel('abs(I)')     
    grid on; xlim([min(zarg), max(zarg)])        
 
    subplot(plotNos.total, 1, plotNos.screen)
    psiValDirect = buildPsiVals(kPsi, psiParams.compl_ampls, zarg);    
    
    yyaxis left      
    hold on; hPsi = plot(zarg, psiValDirect, 'm-', 'linewidth', 8, 'displayname', '\Psi');  
    hold on; hPsi_stored = plot(storedPsi.arg, storedPsi.val, 'k-', 'linewidth', 2, 'displayname', 'stored \Psi');     
    hold on; plot(zarg, zeros(size(zarg)), 'b-', 'linewidth', 8, 'displayname', '\Psi^{rec} = 0')      
    title('Screen density')
      
    yyaxis right
    zarg_dd = zarg(2:end-1);     
    psiVal_dd = diff(diff(psiValDirect)) / (setup.steps.I)^2; 
    hold on; hPsi_dd = plot(zarg_dd, psiVal_dd, 'm:', 'linewidth', 8, 'displayname', '\Psi''''');    
    hold on; hPsi_dd_stored = plot(storedPsi.arg, storedPsi_dd_Val, 'k:', 'linewidth', 2, 'displayname', 'stored \Psi'''''); 
    centerYlim
    ylabel('\Psi''''(s)')
 
    hl = legend([hPsi, hPsi_dd, hPsi_stored, hPsi_dd_stored], ... 
                {'\Psi', '\Psi''''', '\Psi (stored)', '\Psi'''' (stored)'}); 
    set(hl, 'autoupdate', 'off')

    yyaxis left
    ylabel('\Psi(s)')
    yl = ylim; maxabs_yl = max(abs(yl));     
    ylim([-maxabs_yl, maxabs_yl]); 
    grid on; xlim([min(zarg), max(zarg)])       
    
    g = gca;     
    g.YAxis(1).Color = 'k'; 
    g.YAxis(2).Color = 'k';       
end    

% see psiFunFourier -> psiValFun
function val = buildPsiVals(kPsi, compl_ampls, x)
    val = zeros(size(x)); 
    for ik = 1 : numel(compl_ampls)
        val = val + real(compl_ampls(ik) * exp(1i * kPsi(ik) * x)); 
    end  
end


function buildAndPlotImageCurve(clr, lw, uscStruct, psiParams, setup, isCompensate, dn_Ipsi)
    imgStruct.withPsi_raw = BareboneUtilsIonoAF.signal2img_window(uscStruct, psiParams, setup, isCompensate);  
    plotImageCurve(clr, lw, imgStruct.withPsi_raw.yarg, imgStruct.withPsi_raw.complVal, dn_Ipsi); 
end

function plotImageCurve(clr, lw, yarg, complVal, dn_Ipsi)    
    absI = abs(complVal);
    
    hl = findobj(gcf, 'Type', 'Legend');
    if numel(dn_Ipsi) > 0    
        set(hl, 'AutoUpdat', 'On')
        hold on; plot(yarg, absI, [clr '-'], 'linewidth', lw, 'displayname', dn_Ipsi); 
        set(hl, 'AutoUpdat', 'Off')
    else
        hold on; plot(yarg, absI, [clr '-'], 'linewidth', lw); 
    end
end

%{
Relation to the document "SAR Autofocus using Machine Learning", as of 08/05/2023. 

For all records: 

X -> dataset.meta.X --- coordinates of antenna (<--> U)
Z -> dataset.meta.Z --- coordinates of antenna (<--> mu, Xi)
S -> dataset.meta.S --- coordinates of phase screen (<--> Theta, Theta'') 
F -> dataset.meta.setup.F --- dimensionless length of the synthetic aperture
xi -> dataset.meta.setup.xi --- relative screen elevation, should be equal to 0.5

For iseed = 1,2,3,4:

mu      -> dataset.records{iseed}.nuStructs.withSpeckle.complVal    - total target reflectivity (point scatterers + speckle)
Xi      -> dataset.records{iseed}.nuStructs.withoutSpeckle.complVal - target reflectivity, point scatterers only  
Theta   -> dataset.records{iseed}.storedPsi.val                     - screen density 
Theta'' -> dataset.records{iseed}.storedPsi_dd_Val                  - the second derivative of the screen density 
U       -> dataset.records{iseed}.uscStruct.vals                    - data (the received antenna signal)

%}