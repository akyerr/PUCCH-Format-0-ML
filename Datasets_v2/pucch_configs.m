clc
clear
close all

bit_len_harq = 1;
bit_len_sr = 1;


PUCCH_RBs = 0:20:220;
PUCCH_symbs = 0:13;

cfg_count = 0;
for s = 1: length(PUCCH_symbs)
    for rb = 1: length(PUCCH_RBs)
        cfg_count = cfg_count + 1;
        pucch_cfg(cfg_count) = nrPUCCH0Config;
        pucch_cfg(cfg_count).NSizeBWP = 273;
        pucch_cfg(cfg_count).NStartBWP = 0;
        pucch_cfg(cfg_count).FrequencyHopping = 'neither';

        pucch_cfg(cfg_count).InitialCyclicShift = rb-1; % m0
        pucch_cfg(cfg_count).SymbolAllocation = [PUCCH_symbs(s), 1]; %Time alloc:  start symbol: s-1, 1 symbol
        pucch_cfg(cfg_count).PRBSet = PUCCH_RBs(rb); % Freq allocation
    end
end

num_pucch_configs = cfg_count;



