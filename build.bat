@echo off

:: if the vulkan bindings are not generated, do:
::   create folders vulkan/ and vulkan/c
::   run "python create_vulkan_odin_wrapper.py"

:: you need to have glslc.exe on your PATH. this comes with the Vulkan SDK.
pushd build
glslc ..\shaders\shader.vert -o vert.spv
glslc ..\shaders\shader.frag -o frag.spv
odin build ..\main.odin -debug
popd

:: F8 to compile in my vim
:: odin command cheatsheet:
:: odin run main.odin
:: odin build main.odin
:: -debug
::   Enabled debug information, and defines the global constant ODIN_DEBUG to be 'true'
:: -opt:<integer>
::         Set the optimization level for complication
::         Accepted values: 0, 1, 2, 3
::         Example: -opt:2

