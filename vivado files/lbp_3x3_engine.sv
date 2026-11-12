// ============================================================================
// lbp_3x3_engine.sv
// Streaming 3x3 Local Binary Pattern engine (256x256, 8-bit gray, raster order)
//
// Matches reference_model.ipynb :: lbp_calculated_pixel()
//   bit0=TL bit1=T bit2=TR bit3=R bit4=BR bit5=B bit6=BL bit7=L
//   bit = (neighbour >= centre); neighbour outside the image -> bit = 0
//
// One step (en=1) consumes one pixel. The centre pixel of the window lags the
// input by 257 steps (one full line + 1), so a frame needs 65536 + 257 steps:
// the extra 257 are flush steps (pix_in = 0) driven by lbp_fsm_ctrl.
// code_valid is a registered 1-cycle pulse that follows each step once the
// pipeline has filled.
// ============================================================================
module lbp_3x3_engine (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       frame_start,   // 1-cycle pulse: reset counters for new frame
    input  logic       en,            // advance one step
    input  logic [7:0] pix_in,        // next pixel (0 during flush)

    output logic       code_valid,    // 1-cycle pulse, one per centre pixel
    output logic [7:0] lbp_code,
    output logic [3:0] blk_idx,       // {row[7:6], col[7:6]}  -> 4x4 grid, 64x64 blocks
    output logic       code_last      // with code_valid on centre (255,255)
);
    // ---- line buffers (two lines, circular pointer) ------------------------
    logic [7:0] lb1 [0:255];
    logic [7:0] lb2 [0:255];
    logic [7:0] ptr;

    wire [7:0] lb1_rd = lb1[ptr];     // pixel from 256 steps ago
    wire [7:0] lb2_rd = lb2[ptr];     // pixel from 512 steps ago

    // ---- 3x3 window: w[row][col], row0=top, col0=left, col2 = newest -------
    logic [7:0] w [0:2][0:2];
    logic [7:0] wn [0:2][0:2];

    always_comb begin
        for (int r = 0; r < 3; r++) begin
            wn[r][0] = w[r][1];
            wn[r][1] = w[r][2];
        end
        wn[0][2] = lb2_rd;
        wn[1][2] = lb1_rd;
        wn[2][2] = pix_in;
    end

    // ---- centre coordinates of the window that wn will become --------------
    logic [8:0] fill_cnt;             // counts to 257 (fill latency)
    logic [7:0] n_row, n_col;
    wire        filled  = (fill_cnt == 9'd257);
    wire        v_now   = en && filled;

    // out-of-image masks for the new window
    wire top_ok   = (n_row != 8'd0);
    wire bot_ok   = (n_row != 8'd255);
    wire left_ok  = (n_col != 8'd0);
    wire right_ok = (n_col != 8'd255);

    wire [7:0] c = wn[1][1];
    logic [7:0] code_n;
    always_comb begin
        code_n[0] = top_ok && left_ok  && (wn[0][0] >= c);   // TL
        code_n[1] = top_ok             && (wn[0][1] >= c);   // T
        code_n[2] = top_ok && right_ok && (wn[0][2] >= c);   // TR
        code_n[3] =           right_ok && (wn[1][2] >= c);   // R
        code_n[4] = bot_ok && right_ok && (wn[2][2] >= c);   // BR
        code_n[5] = bot_ok             && (wn[2][1] >= c);   // B
        code_n[6] = bot_ok && left_ok  && (wn[2][0] >= c);   // BL
        code_n[7] =           left_ok  && (wn[1][0] >= c);   // L
    end

    always_ff @(posedge clk) begin
        if (en) begin
            lb1[ptr] <= pix_in;
            lb2[ptr] <= lb1_rd;
            for (int r = 0; r < 3; r++)
                for (int k = 0; k < 3; k++)
                    w[r][k] <= wn[r][k];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ptr        <= '0;
            fill_cnt   <= '0;
            n_row      <= '0;
            n_col      <= '0;
            code_valid <= 1'b0;
            lbp_code   <= '0;
            blk_idx    <= '0;
            code_last  <= 1'b0;
        end else begin
            code_valid <= 1'b0;
            code_last  <= 1'b0;

            if (frame_start) begin
                ptr      <= '0;
                fill_cnt <= '0;
                n_row    <= '0;
                n_col    <= '0;
            end else if (en) begin
                ptr <= ptr + 8'd1;
                if (!filled) fill_cnt <= fill_cnt + 9'd1;

                if (v_now) begin
                    code_valid <= 1'b1;
                    lbp_code   <= code_n;
                    blk_idx    <= {n_row[7:6], n_col[7:6]};
                    code_last  <= (n_row == 8'd255) && (n_col == 8'd255);
                    n_col      <= n_col + 8'd1;          // wraps 255 -> 0
                    if (n_col == 8'd255) n_row <= n_row + 8'd1;
                end
            end
        end
    end
endmodule
