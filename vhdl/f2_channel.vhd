-- This is f2_channel VHDL model
-- Generated by initentity script
LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.numeric_std.all;
USE std.textio.all;


ENTITY f2_channel IS
    PORT( A : IN  STD_LOGIC;
          Z : OUT STD_LOGIC
        );
END f2_channel;

ARCHITECTURE rtl OF f2_channel IS
BEGIN
    buf:PROCESS(A)
    BEGIN
        Z<=A;
    END PROCESS;
END ARCHITECTURE;
