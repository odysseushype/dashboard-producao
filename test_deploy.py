import subprocess
import sys
import time
import requests

def test_local():
    """Testa a aplicação localmente antes do deploy"""
    print("🧪 Testando aplicação localmente...")
    
    try:
        # Verificar se consegue importar o módulo
        import relatorio
        print("✅ Módulo relatorio importado com sucesso")
        
        # Verificar se tem server
        if hasattr(relatorio, 'server'):
            print("✅ Servidor Flask configurado")
        else:
            print("❌ Servidor Flask não encontrado")
            return False
            
        print("✅ Aplicação configurada corretamente para deploy")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar aplicação: {e}")
        return False

if __name__ == "__main__":
    if test_local():
        print("\n🟢 Aplicação pronta para deploy!")
    else:
        print("\n🔴 Corrija os erros antes do deploy")