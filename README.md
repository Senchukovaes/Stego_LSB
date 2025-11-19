# Лабораторная 2: стеганография в изображения (LSB)

## Инструкция по запуску
Для запуска встраивания/восстановления сообщения необходимо создать конфигурацию запуска.

В разделе **script** нужно указать модуль, обрабатывающий входные команды и запускающий шифровку/дешифровку - stegopic.py. В разделе **script parametrs** нужно ввести саму команду (они указаны ниже). В разделе **Working directory** необходимо указать рабочую директорию, это, непосредственно, сама Stego_LSB.

## Команды запуска
### Встраивание сообщения
> Значение **payload** в конце команды можно поменять или не указывать вообще (тогда по умолчанию оно будет равно 0.005)
```
embed --in imgs/original/squirrel.png --out imgs/steg/squirrel_steg.png --message.file message.txt --payload 0.005
```
```
embed --in imgs/original/noise_texture.png --out imgs/steg/noise_texture_steg.png --message.file message.txt --payload 0.005
```
```
embed --in imgs/original/gradient.png --out imgs/steg/gradient_steg.png --message.file message.txt --payload 0.005
```
```
embed --in imgs/original/checkerboard.png --out imgs/steg/checkerboard_steg.png --message.file message.txt --payload 0.005
```

### Восстановление сообщения
```
extract --in imgs/steg/squirrel_steg.png --outdir results
```
```
extract --in imgs/steg/noise_texture_steg.png --outdir results
```
```
extract --in imgs/steg/gradient_steg.png --outdir results
```
```
extract --in imgs/steg/checkerboard_steg.png --outdir results
```
