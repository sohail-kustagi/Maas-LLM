import sys

def patch_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    in_func = False
    new_lines = []
    patched = False

    for i, line in enumerate(lines):
        if "void ggml_gemv_q4_K_8x8_q8_K(" in line:
            in_func = True
            
        if in_func and "for (int b = 0; b < nb; b++) {" in line and not patched:
            pragmas = """
#if defined(__clang__)
#pragma clang loop vectorize(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#pragma GCC unroll 4
#endif
"""
            new_lines.append(pragmas)
            patched = True
            
        if in_func and line.startswith("}"):
            # We reached the end of the function, but wait, there might be nested braces
            # We already patched it, so it's fine
            pass
            
        new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)

    if patched:
        print("Successfully patched SVE pragmas into ggml_gemv_q4_K_8x8_q8_K")
    else:
        print("Failed to find target function or loop")

if __name__ == "__main__":
    patch_file(sys.argv[1])
