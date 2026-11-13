// =============================================================================
// sync_2ff.sv  --  two-flop synchronizer for multi-bit GRAY-coded buses
//
// Only safe for buses where at most ONE bit changes per source clock edge
// (Gray pointers). Never use this for arbitrary binary data.
//
// ASYNC_REG = "TRUE" tells Vivado:
//   * place both stages in the same slice (minimum routing between them),
//   * don't optimise / retime / replicate them,
//   * treat them as synchronizers in report_cdc / MTBF analysis.
// =============================================================================

module sync_2ff #(
  parameter int WIDTH = 1
)(
  input  logic             clk,
  input  logic             rst_n,
  input  logic [WIDTH-1:0] d,
  output logic [WIDTH-1:0] q
);

  (* ASYNC_REG = "TRUE" *) logic [WIDTH-1:0] stage1;
  (* ASYNC_REG = "TRUE" *) logic [WIDTH-1:0] stage2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stage1 <= '0;
      stage2 <= '0;
    end else begin
      stage1 <= d;        // may go metastable; has one full cycle to resolve
      stage2 <= stage1;
    end
  end

  assign q = stage2;

endmodule