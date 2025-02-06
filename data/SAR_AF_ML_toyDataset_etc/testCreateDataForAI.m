% modified from minusI4_BareboneSetup
function testCreateDataForAI

    % data generation size
    seeds.count = 10000;

    stepRefinePow = 2;  
    ionoNharm = 6; 
    seeds.start = struct('ionosphere', 21, 'clutter', 61, 'PS', 41); 
    outputMatfname = 'radarSeries.mat'; 
    createDataForRangeOfSeeds(stepRefinePow, ionoNharm, seeds, outputMatfname); 
        
    disp 'DONE'
end

function [setup, init_rng_seed, initHarmonicIdx] = createSetupForAI(stepRefinePow, ionoNharm)    
    setup.F = 100; 
    setup.ionoNharm = ionoNharm; 
    initHarmonicIdx = 1:ionoNharm;    

    init_rng_seed.ionosphere = 36; 
    init_rng_seed.clutter = 31; 
    init_rng_seed.PS = 33;     
    
    setup.domainLengthInF = 3.6; 
    setup.maxN_PS = 15; 1; 5; 
    setup.PoissonLambda_amplPS = 5; 
    setup.numPsiSamples = 7; 

    refinedStep = 1/(2^stepRefinePow);     
    setup.steps.I   = refinedStep; 
    setup.steps.usc = refinedStep;
    setup.steps.nu  = refinedStep;     
 
    setup.xi = 0.5; 
    setup.relNoiseCoeff = 0.05; % 0.4
    setup.addSpeckleCoeff = 0.05; % 0.4
    setup.minScattererRadius = 1;
  
    % see createSetup_etc in minusI4_BareboneSetup, option  'reconstr_shortscale_rect'
    setup.ionoAmplOverPi = 4; 5; 6; 8; 4; 2; 
    setup.F_to_lmax = 1.5; 
    
    setup.windowFun = @(x) ones(size(x));                    setup.windowType = 'rect'; 
    %setup.windowFun = @BareboneUtilsIonoAF.parabolicWindow; setup.windowType = 'parabolic'; 

    % setup.createPsiImplFun = @createFourierPsiWithInterpolant; 
    setup.createPsiImplFun = @createFourierPsiWithCache; % PREFERRED for xi = 0.5  
  
    %setup.sumFun = @sum;                           setup.sumType = 'plain';
    setup.sumFun = @BareboneUtilsIonoAF.sumTrapz;   setup.sumType = 'trapz';     
end

function createDataForRangeOfSeeds(stepRefinePow, ionoNharm, seeds, outputMatfname) 
    [setup, init_rng_seed, initHarmonicIdx] = createSetupForAI(stepRefinePow, ionoNharm); 
    
    dataset.records{seeds.count} = nan; 

    dataset.meta.kPsi = BareboneUtilsIonoAF.get_kPsi_relToF(setup); 
    dataset.meta.initHarmonicIdx = initHarmonicIdx; 
    dataset.meta.init_rng_seed = init_rng_seed; 
    dataset.meta.setup = setup; 
    
    for iseed = 1 : seeds.count
        %isOutput = (rem(iseed, 200) == 1); 
        isOutput = true; 
        if isOutput
            fprintf('Data generation: total seeds: %d, increment no. %d', seeds.count, iseed); 
        end
      
        rng_seed = replaceRngSeed(init_rng_seed, 'ionosphere', seeds.start, iseed, isOutput); 
        rng_seed = replaceRngSeed(     rng_seed, 'clutter',  seeds.start, iseed, isOutput); 
        rng_seed = replaceRngSeed(     rng_seed, 'PS',  seeds.start, iseed, isOutput);       
        setup.rng_seed = rng_seed; 
 
        rec.iseed = iseed; 
        setup.rng_seed.counter = iseed;      
        
        [rec.uscStruct, rec.psiParams, rec.PSstruct, rec.nuStructs] ... 
            = build_uscStruct_psiParams_PSstruct_fromSetup(setup, initHarmonicIdx);
        rec.storedPsi = rec.psiParams.getStoredPsi(); 
        rec.storedPsi_dd_Val = buildPsi_dd_Vals(dataset.meta.kPsi, rec.psiParams.compl_ampls, rec.storedPsi.arg);         
        dataset.records{iseed} = rec; 

        %% Added this section to store compl_ampls
        % Store compl_ampls to dataset
        compl_ampls = rec.psiParams.compl_ampls;    % <-- Extract the compl_ampls
        dataset.compl_ampls{iseed} = compl_ampls;          % <-- Store compl_ampls for each seed
        %% End of added section
     
        if iseed == 1 % lazy init 
            dataset.meta.X = rec.uscStruct.xarg;           % antenna coords % do NOT produce image, so dont need to store image coords            
            dataset.meta.Z = rec.nuStructs.withSpeckle.zarg; % scatterer coords
            dataset.meta.S = rec.storedPsi.arg;            % screen coords
        else % sanity check 
            assert(isequal(dataset.meta.X, rec.uscStruct.xarg));
            assert(isequal(dataset.meta.Z, rec.nuStructs.withSpeckle.zarg)); 
            assert(isequal(dataset.meta.S, rec.storedPsi.arg));              
        end
        
        if isOutput
            reportPSstruct(rec.PSstruct); 
            fprintf('\n')      
        end        
    end
    
    save(outputMatfname, 'dataset','-v7.3');
end

function reportPSstruct(PSstruct)
    nPS = numel(PSstruct.locs); 
    if nPS == 0
        fprintf(': --- NO scatterers');
    else
        fprintf(': %d scatterers\n', nPS);
        for iPS = 1:nPS
            fprintf('           '); 
            fprintf('no. %d: coord = %g, ampl (re, im;  abs) = (%.3f, %.3f;   %.3f) \n', ... 
                     iPS, PSstruct.locs(iPS), real(PSstruct.ampl(iPS)), imag(PSstruct.ampl(iPS)), abs(PSstruct.ampl(iPS)));     
        end 
    end
end

% see psiFunFourier -> psiValFun
function val = buildPsi_dd_Vals(kPsi, compl_ampls, x)
    val = zeros(size(x)); 
    for ik = 1 : numel(compl_ampls)
        val = val - kPsi(ik)^2 * real(compl_ampls(ik) * exp(1i * kPsi(ik) * x)); 
    end  
end

function [uscStruct, psiParams, PSstruct, nuStructs] ... 
         = build_uscStruct_psiParams_PSstruct_fromSetup(setup, initHarmonicIdx)

    assert(rem(setup.F, 1) == 0);    
    
    [psCoord, psAmpl] = create_psCoord_psAmpl(setup); 
    
    PSstruct = BareboneUtilsIonoAF.createPSstruct(psCoord, psAmpl, setup);

    zarg = BareboneUtilsIonoAF.create_zarg(setup); 
    nuStructs.withoutSpeckle = BareboneUtilsIonoAF.PSstruct2nuStruct_assertMatch(PSstruct, zarg);          
    nuStructs.withSpeckle = addSpeckleToNuStruct(nuStructs.withoutSpeckle, setup); 
    
    init_compl_ampls = createCustomPsiComplAmpls(setup, initHarmonicIdx); 
    psiParams = setup.createPsiImplFun(init_compl_ampls, setup); 
    
    uscStruct = BareboneUtilsIonoAF.scene2signal_window(nuStructs.withSpeckle, psiParams, setup);   
    
    uscAbsMax = max(abs(uscStruct.vals(:))); 
    sizeVals = size(uscStruct.vals); 
    complRandom = (randn(sizeVals) + 1i * randn(sizeVals)) / sqrt(2); 
    uscStruct.vals = uscStruct.vals + setup.relNoiseCoeff * uscAbsMax * complRandom;     
end

function [psCoord, psAmpl] = create_psCoord_psAmpl(setup)
    rng(setup.rng_seed.PS);  
    
    numPS = randi(setup.maxN_PS + 1) - 1; % we allow it to be 0 PSs
    if numPS == 0
        psCoord = []; 
        psAmpl = []; 
        return; 
    end
    
    % restrict PScoord to integer values, don't allow repeat coords
    speckleDomainLims = BareboneUtilsIonoAF.getSpeckleDomainLims(setup); 
    imgDomainLims = speckleDomainLims + setup.F * [1, -1]; 
    assert( (rem(imgDomainLims(1), 1) == 0) && ...
            (rem(imgDomainLims(2), 1) == 0) ); 
    idxCountWithinDomain = diff(imgDomainLims) + 1; 

    psCoord = zeros(1, numPS);
    minRadius = setup.minScattererRadius; % Minimum radius distance
    
    for i = 1:numPS
        isValid = false;
        while ~isValid
            candidate = imgDomainLims(1) - 1 + randi(idxCountWithinDomain);
            if i == 1 || all(abs(candidate - psCoord(1:i-1)) >= minRadius)
                psCoord(i) = candidate;
                isValid = true;
            end
        end
    end
    
    % normalize amplitude by 1 here
    if setup.PoissonLambda_amplPS == inf
        psAmpl = ones(size(psCoord));
    else
        PoissonLambda = setup.PoissonLambda_amplPS; 
        amplAbs = poissrnd(PoissonLambda, [numPS, 1]) / PoissonLambda; % normalize
        phase = 2 * pi * rand([numPS, 1]); 
        psAmpl = amplAbs .* exp(1i * phase); 
    end
end

function nuStructWithSpeckle = addSpeckleToNuStruct(nuStruct, setup)
    rng(setup.rng_seed.clutter);      
    speckleRe = randn(size(nuStruct.zarg)); 
    speckleIm = randn(size(nuStruct.zarg));    

    speckle_complVal = (speckleRe + 1i * speckleIm) / sqrt(2); 

    nuStructWithSpeckle.zarg = nuStruct.zarg; 
    speckleStatCoeff = setup.addSpeckleCoeff * sqrt(setup.steps.nu); % the additional sqrt is to account for incoherent summation of speckle contribution in one resolution cell 
    nuStructWithSpeckle.complVal = nuStruct.complVal + speckleStatCoeff * speckle_complVal; 
end

function rng_seed = replaceRngSeed(rng_seed_in, rng_fieldName, seeds_start, iseed, isOutput)
    rng_seed = rng_seed_in; 
    seed = seeds_start.(rng_fieldName) + iseed - 1; % 

    rng_seed = rmfield(rng_seed, rng_fieldName); % for safety; 
    rng_seed.(rng_fieldName) = seed; 
    
    if isOutput
        fprintf(', %s --- %d', rng_fieldName, seed);
    end
end

function psiParams = createFourierPsiWithCache(compl_ampls, setup)
    psiParams = psiFunFourierWithCache(compl_ampls); 
    psiParams.initializeCache(setup);
end

function compl_ampls = createCustomPsiComplAmpls(setup, initHarmonicIdx)
    rng(setup.rng_seed.ionosphere);        
    phases = 2 * pi * rand(setup.ionoNharm, 1); 
    abs_ampls = createScaledAmplsAbs(setup, initHarmonicIdx);
    compl_ampls = abs_ampls .* exp(1i * phases);
end

function abs_ampls = createScaledAmplsAbs(setup, initHarmonicIdx)
    ampls_norm = createCustomAmplsNorm(setup, initHarmonicIdx); 
    abs_ampls = setup.ionoAmplOverPi * pi * ampls_norm; 
end

function ampls_norm = createCustomAmplsNorm(setup, initHarmonicIdx)
    kPsi = BareboneUtilsIonoAF.get_kPsi_relToF(setup); 

    ampls_unnorm = zeros(size(kPsi));
    ampls_unnorm(initHarmonicIdx) = kPsi(initHarmonicIdx).^(-2);      % !!! NOTE: this is where we define the spectrum for psi
    energy_unnorm = sum(ampls_unnorm.^2); 
    ampls_norm = ampls_unnorm / sqrt(energy_unnorm); % should be <= 1
end  
