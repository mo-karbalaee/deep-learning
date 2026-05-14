import ex0.src_to_implement.pattern as pattern
import ex0.src_to_implement.generator as generator

def main():
    c = pattern.Checker(250, 25)
    c.show()
    
    circ = pattern.Circle(1024, 200, (512, 256))
    circ.show()
    
    spec = pattern.Spectrum(255)
    spec.show()
    
    gen = generator.ImageGenerator('exercise_data/', 'Labels.json', 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
    gen.show()

if __name__ == '__main__':
    main()
