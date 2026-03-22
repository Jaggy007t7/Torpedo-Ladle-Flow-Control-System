`timescale 1ns/1ps
`include "include.v"

// Weight Memory;
module Weight_Memory #(parameter numWeight = 3, neuronNo=5,layerNo=1,addressWidth=10,dataWidth=16,weightFile="w_1_15.mif") 
    ( 
    input clk,
    input wen,
    input ren,
    input [addressWidth-1:0] wadd,  //address size is 
    input [addressWidth-1:0] radd,
    input [dataWidth-1:0] win,
    output reg [dataWidth-1:0] wout);  //16 bit no.
    
    reg [dataWidth-1:0] mem [0:numWeight-1];

    `ifdef pretrained
        initial
		begin
	        $readmemb(weightFile, mem,0, numWeight-1);
	    end
	`else
		always @(posedge clk)
		begin
			if (wen)
			begin
				mem[wadd] <= win;
			end
		end 
    `endif
    
    always @(posedge clk)
    begin
        if (ren)
        begin
            wout <= mem[radd];
        end
    end 
endmodule

// Sigmoid ROM;
module Sig_ROM #(parameter inWidth=10, dataWidth=16) (
    input           clk,
    input   [inWidth-1:0]   x,
    output  [dataWidth-1:0]  out
    );
    
    reg [dataWidth-1:0] mem [2**inWidth-1:0];
    reg [inWidth-1:0] y;
	
	initial
	begin
		$readmemb("sigContent.mif",mem);
	end
    
    always @(posedge clk)
    begin
        if($signed(x) >= 0)
            y <= x+(2**(inWidth-1));
        else 
            y <= x-(2**(inWidth-1));      
    end
    
    assign out = mem[y];
    
endmodule

// ReLU;
module ReLU  #(parameter dataWidth=16,weightIntWidth=4) (
    input           clk,
    input   [2*dataWidth-1:0]   x,
    output  reg [dataWidth-1:0]  out
);


always @(posedge clk)
begin
    if($signed(x) >= 0) //If no. is positive.
    begin
        if(x[2*dataWidth-1-:weightIntWidth+1]) //over flow to sign bit of integer part. Problem at the logic.
            out <= {1'b0,{(dataWidth-1){1'b1}}}; //positive saturate
        else
            out <= x[2*dataWidth-1-weightIntWidth-:dataWidth];
    end
    else //If no. is negative.
        out <= 0;      
end

endmodule



// Main Top Module;
module neuron #(parameter layerNo=0,neuronNo=0,numWeight=784,dataWidth=16,sigmoidSize=10,weightIntWidth=4,actType="relu",biasFile="b_1_15.mif",weightFile="w_1_15.mif")(
    input           clk,
    input           rst,
    input [dataWidth-1:0]    myinput,
    input           myinputValid,
    input           weightValid,
    input           biasValid,
    input [31:0]    weightValue,
    input [31:0]    biasValue,
    input [31:0]    config_layer_num,
    input [31:0]    config_neuron_num,
    output[dataWidth-1:0]    out,
    output reg      outvalid   
    );
    
    parameter addressWidth = $clog2(numWeight);
    
    reg         wen;
    wire        ren;
    reg [addressWidth-1:0] w_addr;
    reg [addressWidth-1:0]   r_addr;//read address has to reach until numWeight hence width is 1 bit more
    reg [dataWidth-1:0]  w_in;
    wire [dataWidth-1:0] w_out;
    reg [2*dataWidth-1:0]  mul; 
    reg [2*dataWidth-1:0]  sum;
    reg [2*dataWidth-1:0]  bias;
    reg [31:0]    biasReg[0:0];
    reg         weight_valid;
    reg         mult_valid;
    wire        mux_valid;
    reg         sigValid; 
    wire [2*dataWidth:0] comboAdd;
    wire [2*dataWidth:0] BiasAdd;
    reg  [dataWidth-1:0] myinputd;
    reg muxValid_d;
    reg muxValid_f;
    reg addr=0;
   //Loading weight values into the momory
    always @(posedge clk)
    begin
        if(rst)
        begin
            w_addr <= {addressWidth{1'b1}};
            wen <=0;
        end
        else if(weightValid & (config_layer_num==layerNo) & (config_neuron_num==neuronNo))
        begin
            w_in <= weightValue;
            w_addr <= w_addr + 1;
            wen <= 1;
        end
        else
            wen <= 0;
    end
	
    assign mux_valid = mult_valid;
    assign comboAdd = mul + sum;
    assign BiasAdd = bias + sum;
    assign ren = myinputValid;
    
	`ifdef pretrained
		initial
		begin
			$readmemb(biasFile,biasReg);
		end
		always @(posedge clk)
		begin
            bias <= {biasReg[addr][dataWidth-1:0],{dataWidth{1'b0}}};
        end
	`else
		always @(posedge clk)
		begin
			if(biasValid & (config_layer_num==layerNo) & (config_neuron_num==neuronNo))
			begin
				bias <= {biasValue[dataWidth-1:0],{dataWidth{1'b0}}};
			end
		end
	`endif
    
    
    always @(posedge clk)
    begin
        if(rst|outvalid)
            r_addr <= 0;
        else if(myinputValid)
            r_addr <= r_addr + 1;
    end
    
    always @(posedge clk)
    begin
        mul  <= $signed(myinputd) * $signed(w_out);
    end
    
    
    always @(posedge clk)
    begin
        if(rst|outvalid)
            sum <= 0;
        else if((r_addr == numWeight) & muxValid_f)
        begin
            if(!bias[2*dataWidth-1] &!sum[2*dataWidth-1] & BiasAdd[2*dataWidth-1]) //If bias and sum are positive and after adding bias to sum, if sign bit becomes 1, saturate
            begin
                sum[2*dataWidth-1] <= 1'b0;
                sum[2*dataWidth-2:0] <= {2*dataWidth-1{1'b1}};
            end
            else if(bias[2*dataWidth-1] & sum[2*dataWidth-1] &  !BiasAdd[2*dataWidth-1]) //If bias and sum are negative and after addition if sign bit is 0, saturate
            begin
                sum[2*dataWidth-1] <= 1'b1;
                sum[2*dataWidth-2:0] <= {2*dataWidth-1{1'b0}};
            end
            else
                sum <= BiasAdd; 
        end
        else if(mux_valid)
        begin
            if(!mul[2*dataWidth-1] & !sum[2*dataWidth-1] & comboAdd[2*dataWidth-1])
            begin
                sum[2*dataWidth-1] <= 1'b0;
                sum[2*dataWidth-2:0] <= {2*dataWidth-1{1'b1}};
            end
            else if(mul[2*dataWidth-1] & sum[2*dataWidth-1] & !comboAdd[2*dataWidth-1])
            begin
                sum[2*dataWidth-1] <= 1'b1;
                sum[2*dataWidth-2:0] <= {2*dataWidth-1{1'b0}};
            end
            else
                sum <= comboAdd; 
        end
    end
    
    always @(posedge clk)
    begin
        myinputd <= myinput;
        weight_valid <= myinputValid;
        mult_valid <= weight_valid;
        sigValid <= ((r_addr == numWeight) & muxValid_f) ? 1'b1 : 1'b0;
        outvalid <= sigValid;
        muxValid_d <= mux_valid;
        muxValid_f <= !mux_valid & muxValid_d;
    end
    
    
    //Instantiation of Memory for Weights
    Weight_Memory #(.numWeight(numWeight),.neuronNo(neuronNo),.layerNo(layerNo),.addressWidth(addressWidth),.dataWidth(dataWidth),.weightFile(weightFile)) WM(
        .clk(clk),
        .wen(wen),
        .ren(ren),
        .wadd(w_addr),
        .radd(r_addr),
        .win(w_in),
        .wout(w_out)
    );
    
    generate
        if(actType == "sigmoid")
        begin:siginst
        //Instantiation of ROM for sigmoid.  (Generation of sigmoid function)
            Sig_ROM #(.inWidth(sigmoidSize),.dataWidth(dataWidth)) s1(
            .clk(clk),
            .x(sum[2*dataWidth-1-:sigmoidSize]),
            .out(out)
        );
        end

        else //relu function generation.
        begin:ReLUinst
            ReLU #(.dataWidth(dataWidth),.weightIntWidth(weightIntWidth)) s1 (
            .clk(clk),
            .x(sum),
            .out(out)
        );
        end
    endgenerate

    `ifdef DEBUG
    always @(posedge clk)
    begin
        if(outvalid)
            $display(neuronNo,,,,"%b",out);
    end
    `endif
endmodule
