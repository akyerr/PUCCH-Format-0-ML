function [ ] = testvectors_text(filename,input,format,flag)
	if flag == false % checking if the debugging is true or not
		return;
	end

	fileID = fopen(filename,'a');
	dim = numel(size(input)); % finding the dimension of input array

	if dim == 3			% Grid with multiple layers
		input = reshape(input,[],size(input,3));
	end

	[m,n] = size(input);
	if format == "complex"
		for i = 1:m
			for j = 1:n
				fprintf(fileID,"%0.5f , %0.5f , " , real(input(i,j)) , imag(input(i,j)));
			end
			fprintf(fileID,"\n");
		end

	elseif format == "bits" || format == "hexa"
		if isreal(input)
			for i = 1:m
				for j = 1:n
					[C] = sub2d(input(i,j),format,1);
					fprintf(fileID,"%s , " , C);
				end
				fprintf(fileID,"\n");
			end
		else
			for i = 1:m
				for j = 1:n
					[C] = sub2d(input(i,j),format,0);
					fprintf(fileID,"%s , " , C);
				end
				fprintf(fileID,"\n");
			end
		end

	else
		for j = 1:n
			if n > 1
				fprintf(fileID,"Codeblock %d\n" , j);
			end
			for i = 1:m
				fprintf(fileID,"%d \n" , input(i,j));
			end
		end
	end
	
	fclose(fileID);
end


function [C] = sub2d(in,format,is_real)

	if is_real == 1
		if format == "bits"
			B = typecast(int16(in*(2^14)),'uint16');
			C = dec2bin(B,16);
		elseif format == "hexa"
			B = typecast(int16(in*(2^14)),'uint16');
			C = dec2hex(B,4);
		end
	else
		if format == "bits"
			Q = imag(in);
			BQ = typecast(int16(Q*(2^14)),'uint16');
			CQ = dec2bin(BQ,16);
			I = real(in);
			BI = typecast(int16(I*(2^14)),'uint16');
			CI = dec2bin(BI,16);
			C = [CQ CI];
		elseif format == "hexa"
			Q = imag(in);
			BQ = typecast(int16(Q*(2^14)),'uint16');
			CQ = dec2hex(BQ,4);
			I = real(in);
			BI = typecast(int16(I*(2^14)),'uint16');
			CI = dec2hex(BI,4);
			C = [CQ CI];
		end
	end
end