"""Apply checked execution hooks to build-directory copies of pinned NumCalc sources."""
from pathlib import Path
import sys


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"NumCalc source changed: expected one occurrence of {old!r}")
    return text.replace(old, new)


def adapt(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob('*.cpp'):
        text = path.read_text()
        if path.name == 'NC_Main.cpp':
            text = '#include "Hooks.h"\n' + text
            text = replace(text, 'int main(int argc, char **argv)', 'int NumCalcAuthorMain(int argc, char **argv)')
            text = replace(text, '    NC_SetupEquationSystem(NCout);', '    NumCalcBeginAssembly();\n    NC_SetupEquationSystem(NCout);')
            begin = text.index('    // solve the equation system\n')
            end = text.index('    // destroy the coefficient matrices\n', begin)
            text = text[:begin] + '    NumCalcNativeSolve(NCout);\n\n' + text[end:]
        elif path.name == 'NC_CommonFunctions.cpp':
            text = '#include "Hooks.h"\n' + text
            text = replace(text, '\t// initialize the solution vector by using the random function',
                           '\tNumCalcResetOperator();\n\t// initialize the solution vector by using the random function')
            text = replace(text, 'err_ori = sqrt(dwk1);',
                           'err_ori = 0.0;\n\t\t\tfor (int row = 0; row < numRowsOfCoefficientMatrix_; ++row) err_ori += zrhs[row].qnorm();\n\t\t\terr_ori = sqrt(err_ori);')
            text = replace(text, '\t\telse\n\t\t{\n            err_rel = sqrt(dwk1)/err_ori;\n\t\t}',
                           '\t\terr_rel = sqrt(dwk1)/err_ori;')
            text = replace(text, 'if(j == niter_max_) {', 'if(j == niter_max_ && !(err_rel < ErroIterSols)) {')
            text = replace(text, 'if(j >= niter_max_ && methodFMM_ == 0)',
                           'if(!(err_rel < ErroIterSols) && methodFMM_ == 0)')
            text = replace(text, '\t\t// Gauss elimination method\n\t\tTfactor_usy(zcoefl, numRowsOfCoefficientMatrix_);\n\t\tTfbelim(zcoefl, zrhs, numRowsOfCoefficientMatrix_);',
                           '\t\tif (methodPreconditioner_ == 0) NC_DeleteIncompleteLUMatrices();\n\t\telse if (methodPreconditioner_ == 1) delete [] dscaling;\n\t\tNumCalcRequestDirectSolve();')
            begin = text.index('\t// compute the product\n\tswitch(methodFMM_)')
            end = text.index('\n// compute the scaling vector', begin)
            text = text[:begin] + '\tNumCalcMultiply(ztmp, zresuvect);\n}\n' + text[end:]
        elif path.name == 'NC_Input.cpp':
            # Leave one iteration for RETU after two boundary records per element.
            text = replace(text, 'maxbcline3 = 2*numElementsOfBoundaryMesh_;',
                           'maxbcline3 = 2*numElementsOfBoundaryMesh_ + 1;')
        elif path.name == 'NC_PostProcessing.cpp':
            # Preserve field values when serializing the native output.
            text = text.replace('% E', '%.17e').replace('%E', '%.17e')
        destination = target / path.name
        if not destination.exists() or destination.read_text() != text:
            destination.write_text(text)


if __name__ == '__main__':
    adapt(Path(sys.argv[1]), Path(sys.argv[2]))
