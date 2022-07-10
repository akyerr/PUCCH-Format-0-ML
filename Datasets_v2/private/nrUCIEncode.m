function codedUCI = nrUCIEncode(uciBits,Euci, folder, format, config_no, varargin)
%nrUCIEncode Uplink control information encoding
%   CODEDUCI = nrUCIEncode(UCIBITS,EUCI) encodes the input UCI bits
%   UCIBITS, as per TS 38.212 Sections 6.3.1.2, 6.3.1.3, 6.3.1.4, and
%   6.3.1.5, to output the concatenated, rate-matched, encoded blocks
%   CODEDUCI, of specified length EUCI for mapping onto PUCCH.
%   Corresponding sections for PUSCH (6.3.2.2, 6.3.2.3, 6.3.2.4, and
%   6.3.2.5) are also covered by the same processing.
%
%   The processing includes code-block segmentation and CRC attachment,
%   channel coding, rate matching and code block concatenation. Both Polar
%   and small block length codes are supported. The function can be used
%   for encoding UCI mapped onto either PUCCH or PUSCH.
%
%   The input UCIBITS must be a binary column vector (double or int8 type)
%   corresponding to the UCI bits and the output is a binary column vector
%   of length EUCI, with same type as input.
%
%   CODEDUCI = nrUCIEncode(UCIBITS,EUCI,MODULATION) encodes the input UCI
%   bits for the modulation scheme MODULATION specified as one of
%   'pi/2-BPSK','QPSK','16QAM','64QAM','256QAM'. MODULATION applies when
%   the length of UCIBITS is 1 or 2 only, and if not specified, defaults to
%   'QPSK'. The place-holder bits "x" and "y" in tables 5.3.3.1-1 and
%   5.3.3.2-1 are represented by "-1" and "-2", respectively for the 1 and
%   2-bit encoding.
%
%   The encoding scheme employed based on the UCIBITS input length, A, is
%   highlighted by the following table:
%     A          Code-block Segmentation   CRC bits    Encoding
%    1                  NA                   NA        Repetition
%    2                  NA                   NA        Simplex
%    3...11             NA                   NA        Reed-Muller
%    12...19            NA                   6         Parity-check Polar
%    20...1706   Conditioned on A,EUCI       11        Polar
%
%   Example 1:
%   % Encode 32 UCI bits for an output length E of 120.
%
%   uci = randi([0 1],32,1,'int8');
%   E = 120;
%   codedUCI = nrUCIEncode(uci,E);
%
%   Example 2:
%   % UCI encode a 2-bit payload to output 12 bits for 16QAM.
%
%   uci = randi([0 1],2,1,'int8');
%   E = 4*3;
%   codedUCI = nrUCIEncode(uci,E,'16QAM');
%
%   See also nrUCIDecode, nrPUCCH2, nrPUCCH3, nrPUCCH4, nrPUSCH,
%   nrPolarEncode, nrRateMatchPolar.

%   Copyright 2018 The MathWorks, Inc.

%#codegen

%   Reference:
%   [1] 3GPP TS 38.212, "3rd Generation Partnership Project; Technical
%   Specification Group Radio Access Network; NR; Multiplexing and channel
%   coding (Release 15). Section 6.3.

%     narginchk(2,3);

    % Empty in, empty out
    typeIn = class(uciBits);
    if isempty(uciBits)
        codedUCI = zeros(0,1,typeIn);
        return;
    end

    % Validate mandatory inputs
    validateInputs(uciBits,Euci);

    A = length(uciBits);
    if A>=12 % Polar coding (both Parity-check and CRC-aided)

        % Code block segmentation, Section 5.2.1
        if (A>=1013) || (A>=360 && Euci>=1088) % Use Euci here
            C = 2;
            Aprime = ceil(A/C)*C;
            paddedUCI = zeros(Aprime,1,typeIn);
            if Aprime~=A
                % prepend filler bit
                paddedUCI(2:end) = uciBits;
            else
                % no filler bit
                paddedUCI = uciBits;
            end
            uciCBs = reshape(paddedUCI,[],C);

            Lcrc = '11';

        else % no segmentation
            C = 1;
            uciCBs = uciBits;

            if A<=19
                Lcrc = '6';
            else
                Lcrc = '11';
            end
        end

        Er = floor(Euci/C);
        % Check Er limits, and feedback in terms of input codeblock length
        coder.internal.errorIf(Er>8192 || Er<=21, ...
            'nr5g:nrUCIEncode:InvalidCBLength',Er);

        nMax = 10;
        iIL = false;
        iBIL = true;

        % CRC attachment, Sections 6.3.1.2.1/6.3.2.2.1, 5.1 [1]
        bitsCRC = nrCRCEncode(uciCBs,Lcrc);
        K = size(bitsCRC,1);

        codedCWs = zeros(Er,C,typeIn);
        for cIdx = 1:C
            % Channel coding, Sections 6.3.1.3.1/6.3.2.3.1, 5.3.1 [1]
            encOut = nrPolarEncode(bitsCRC(:,cIdx),Er,nMax,iIL);
            testvectors_text([folder, 'cb_', num2str(cIdx), '_polar_enc_out_f', num2str(format), '_', num2str(config_no), '.txt'], encOut, 'same', true);
            save([folder, 'cb_', num2str(cIdx), '_polar_enc_out_f', num2str(format), '_', num2str(config_no), '.mat'], 'encOut')
            
            % Rate matching, Sections 6.3.1.4.1/6.3.2.4.1, 5.4.1 [1]
            codedCWs(:,cIdx) = nrRateMatchPolar(encOut,K,Er,iBIL);
            rate_match_out = codedCWs(:,cIdx);
            testvectors_text([folder, 'cb_', num2str(cIdx), '_rate_match_out_f', num2str(format), '_', num2str(config_no), '.txt'], rate_match_out, 'same', true);
            save([folder, 'cb_', num2str(cIdx), '_rate_match_out_f', num2str(format), '_', num2str(config_no), '.mat'], 'rate_match_out')
        end

        % Code block concatenation, Sections 6.3.1.5/6.3.2.5, 5.5 [1]
        G = Er*C + mod(Euci,C);
        codedUCI = zeros(G,1,typeIn);
        codedUCI(1:Er*C,1) = codedCWs(:);

    else % A<12

        % Small block length encoding
        if A<3
            if nargin>5
                % Validate MODULATION, only when needed
                modScheme = varargin{1};
                fcnName = 'nrUCIEncode';
                modlist = {'pi/2-BPSK','QPSK','16QAM','64QAM','256QAM'};
                modulation = validatestring(modScheme,modlist,fcnName,'MODULATION');
            else
                modulation = 'QPSK';
            end

            % Encode 1 or 2 bits, Sections 6.3.1.3.2/6.3.2.3.2,
            % 5.3.3.1/5.3.3.2 [1]
            out = nr5g.internal.smallEncode12(uciBits,modulation);

        else % 3<=A<=11
            % Encode 3...11 bits, Sections 6.3.1.3.2/6.3.2.3.2, 5.3.3.3 [1]
            out = nr5g.internal.smallEncode311(uciBits);
        end
        
        testvectors_text([folder, 'small_enc_out_f', num2str(format), '_', num2str(config_no), '.txt'], out, 'same', true);
        save([folder, 'small_enc_out_f', num2str(format), '_', num2str(config_no), '.mat'], 'out')

        % Rate matching (repetition), Sections 6.3.1.4.2/6.3.2.4.2, 5.4.3 [1]
        N = length(out);
        codedUCI = out(mod(0:Euci-1,N)+1,1);
        
        testvectors_text([folder, 'small_rate_match_out_f', num2str(format), '_', num2str(config_no), '.txt'], codedUCI, 'same', true);
        save([folder, 'small_rate_match_out_f', num2str(format), '_', num2str(config_no), '.mat'], 'codedUCI')
    end

end

function validateInputs(uciBits,Euci)
% Check inputs

    fcnName = 'nrUCIEncode';

    % Validate input UCI message bits
    validateattributes(uciBits,{'int8','double'},{'binary','column'}, ...
        fcnName,'UCIBITS');
    % A must be less than or equal to 1706
    A = length(uciBits);
    coder.internal.errorIf(A>1706,'nr5g:nrUCIEncode:InvalidInputLength',A);

    % Validate rate matched output length which must be greater than minE.
    if A<=11              % Small block lengths
        if A==1
            minE = 0;       % Allows no coding for qm=1
        else
            minE = A;
        end
    elseif A>=12 && A<=19 % PC-Polar
        minE = A+6+3;     % A+crcLen+nPC
    else                  % CA-Polar
        minE = A+11;      % A+crcLen
    end
    validateattributes(Euci,{'numeric'},{'scalar','integer','>',minE}, ...
        fcnName,'E');

end
