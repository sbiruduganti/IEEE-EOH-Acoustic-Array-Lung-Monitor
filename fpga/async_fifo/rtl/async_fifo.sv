// =============================================================================
// async_fifo.sv  --  v1 dual-clock (asynchronous) FIFO
// IEEE EOH 2026-27, FPGA subteam
//
// Architecture: classic Cummings-style async FIFO (SNUG 2002, "Simulation and
// Synthesis Techniques for Asynchronous FIFO Design").
//   * (ADDR_WIDTH+1)-bit pointers: the extra MSB tells "full" apart from "empty"
//     when the address bits are equal.
//   * Pointers cross clock domains as GRAY code, so only one bit changes per
//     increment -> a synchronizer can never capture a wildly wrong value, only
//     the old or the new one.
//   * Two-flop synchronizers (sync_2ff) with ASYNC_REG so Vivado packs them
//     next to each other (maximises metastability settling time / MTBF).
//   * full/empty are computed from the *next* pointer and registered, so they
//     are glitch-free flop outputs.
//   * First-word fall-through (FWFT) read: rd_data already shows the oldest
//     word whenever rd_empty == 0. rd_en means "I consumed it, advance".
//     This maps directly onto AXI4-Stream:
//         s_axis_tready = ~wr_full      wr_en = s_axis_tvalid & s_axis_tready
//         m_axis_tvalid = ~rd_empty     rd_en = m_axis_tvalid & m_axis_tready
//
// Pessimism (safe by construction):
//   * wr_full may stay high a few wr_clk cycles after space frees up, and
//     rd_empty may stay high a few rd_clk cycles after data arrives, because
//     the other side's pointer takes 2 cycles to synchronize. Never the
//     unsafe direction.
//
// Limits of v1 (deliberate):
//   * DEPTH = 2**ADDR_WIDTH (must be a power of two for Gray wrap to work).
//   * ADDR_WIDTH >= 2.
//   * Same width on both sides (no width conversion).
//   * Memory is read asynchronously -> Vivado infers distributed RAM (LUTRAM).
//     Good for shallow CDC FIFOs (16-64 entries). For deep buffers, use a
//     BRAM-based version or xpm_fifo_async.
//   * Reset: assert wr_rst_n and rd_rst_n together, hold each for a few cycles
//     of its own clock. Writing/reading during reset is not allowed.
//
// Timing constraints: see async_fifo.xdc. Without them Vivado reports false
// cross-clock violations and does not bound the Gray-pointer skew.
// =============================================================================

module async_fifo #(
  parameter int DATA_WIDTH = 32,
  parameter int ADDR_WIDTH = 4            // depth = 2**ADDR_WIDTH
)(
  // ---------------- write side (wr_clk domain) ----------------
  input  logic                  wr_clk,
  input  logic                  wr_rst_n,  // active-low, async assert
  input  logic                  wr_en,     // push request (ignored when full)
  input  logic [DATA_WIDTH-1:0] wr_data,
  output logic                  wr_full,

  // ---------------- read side (rd_clk domain) -----------------
  input  logic                  rd_clk,
  input  logic                  rd_rst_n,  // active-low, async assert
  input  logic                  rd_en,     // pop / "consumed" (ignored when empty)
  output logic [DATA_WIDTH-1:0] rd_data,   // valid whenever !rd_empty (FWFT)
  output logic                  rd_empty
);

  localparam int DEPTH = 1 << ADDR_WIDTH;
  localparam int PW    = ADDR_WIDTH + 1;   // pointer width

  // full when write Gray pointer == read Gray pointer with top TWO bits inverted
  localparam logic [PW-1:0] FULL_MASK = {2'b11, {(ADDR_WIDTH-1){1'b0}}};

  // ---------------------------------------------------------------------------
  // Storage
  // ---------------------------------------------------------------------------
  logic [DATA_WIDTH-1:0] mem [DEPTH];

  // ---------------------------------------------------------------------------
  // Write domain
  // ---------------------------------------------------------------------------
  logic [PW-1:0] wbin, wbin_next;      // binary write pointer
  logic [PW-1:0] wgray, wgray_next;    // Gray write pointer (crosses to rd side)
  logic [PW-1:0] rgray_wclk;           // read Gray pointer, synced into wr_clk

  logic wr_fire;
  assign wr_fire    = wr_en & ~wr_full;
  assign wbin_next  = wbin + PW'(wr_fire);
  assign wgray_next = (wbin_next >> 1) ^ wbin_next;

  always_ff @(posedge wr_clk or negedge wr_rst_n) begin
    if (!wr_rst_n) begin
      wbin    <= '0;
      wgray   <= '0;
      wr_full <= 1'b0;
    end else begin
      wbin    <= wbin_next;
      wgray   <= wgray_next;
      wr_full <= (wgray_next == (rgray_wclk ^ FULL_MASK));
    end
  end

  always_ff @(posedge wr_clk) begin
    if (wr_fire)
      mem[wbin[ADDR_WIDTH-1:0]] <= wr_data;
  end

  // ---------------------------------------------------------------------------
  // Read domain
  // ---------------------------------------------------------------------------
  logic [PW-1:0] rbin, rbin_next;      // binary read pointer
  logic [PW-1:0] rgray, rgray_next;    // Gray read pointer (crosses to wr side)
  logic [PW-1:0] wgray_rclk;           // write Gray pointer, synced into rd_clk

  logic rd_fire;
  assign rd_fire    = rd_en & ~rd_empty;
  assign rbin_next  = rbin + PW'(rd_fire);
  assign rgray_next = (rbin_next >> 1) ^ rbin_next;

  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n) begin
      rbin     <= '0;
      rgray    <= '0;
      rd_empty <= 1'b1;
    end else begin
      rbin     <= rbin_next;
      rgray    <= rgray_next;
      rd_empty <= (rgray_next == wgray_rclk);
    end
  end

  // FWFT: head of queue is always presented
  assign rd_data = mem[rbin[ADDR_WIDTH-1:0]];

  // ---------------------------------------------------------------------------
  // Clock-domain crossings: Gray pointers only, each through 2 flops
  // ---------------------------------------------------------------------------
  sync_2ff #(.WIDTH(PW)) u_sync_r2w (
    .clk   (wr_clk),
    .rst_n (wr_rst_n),
    .d     (rgray),
    .q     (rgray_wclk)
  );

  sync_2ff #(.WIDTH(PW)) u_sync_w2r (
    .clk   (rd_clk),
    .rst_n (rd_rst_n),
    .d     (wgray),
    .q     (wgray_rclk)
  );

endmodule