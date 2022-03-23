function estChannelGrid = f2_channel_estimate(carrier, rxGrid, pucchIndices, dmrsIndices, dmrsSymbols)
estChannelGrid = zeros(size(rxGrid));
NRBs = length(dmrsIndices)/4;

dmrsIndRB = [2, 5, 8, 11];
pucchIndRB = [1, 3, 4, 6, 7, 9, 10, 12];
H_dmrs = [];
H_pucch = [];
for r = 1: NRBs
    RB = zeros(12, 1);
    dmrs_vals = dmrsSymbols(length(dmrsIndRB)*(r-1)+1: length(dmrsIndRB)*(r-1) + length(dmrsIndRB));
    rx_dmrs = rxGrid(dmrsIndices(length(dmrsIndRB)*(r-1)+1: length(dmrsIndRB)*(r-1) + length(dmrsIndRB)));
%     disp(length(dmrsIndRB)*(r-1)+1: length(dmrsIndRB)*(r-1) + length(dmrsIndRB))
    
    RB(dmrsIndRB) = rx_dmrs./dmrs_vals;
    
    RB(1) = RB(2);
    RB(3) = (2/3)*RB(2) + (1/3)*RB(5);
    RB(4) = (2/3)*RB(5) + (1/3)*RB(2);
    
    RB(6) = (2/3)*RB(5) + (1/3)*RB(8);
    RB(7) = (2/3)*RB(8) + (1/3)*RB(5);
    
    RB(9) = (2/3)*RB(8) + (1/3)*RB(11);
    RB(10) = (2/3)*RB(11) + (1/3)*RB(8);
    
    RB(12) = RB(11);
    
    H_dmrs = [H_dmrs; RB(dmrsIndRB)]; %#ok
    H_pucch = [H_pucch; RB(pucchIndRB)]; %#ok
    
    dbg = 1;
end
dbg = 1;
estChannelGrid(pucchIndices) = H_pucch;
estChannelGrid(dmrsIndices) = H_dmrs;
end

